# -*- coding: utf-8 -*-
"""
PCH: Projection Convex Hull

Dependencies: numpy, scipy
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
from scipy.special import expit


class PCH:
    """
    Project Convex Hull (PCH)
    Halfspaces: X @ w^T - b >= 0
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.n_splits = args["k_max"]

        # thresholds
        self.threshold = np.ones(self.n_splits) * 100
        self.shift_th = args["shift_th"]
        self.beta = args["beta"]
        self.num_ite_max = args["max_gd_ite"]
        self.max_ite_num = args["max_ite"]

        self.silent = args.get("silent", False)

        # adjacency (exclude itself)
        self.adjacency = np.ones((self.n_splits, self.n_splits)) - np.eye(self.n_splits)

        self.lr = args["learning_rate"]
        self.lr_grad2 = args["weight_lr"]

        # will be set in _initial
        self.w = None
        self.b = None
        self.weight0 = None
        self.grad_cum = None
        self.det_weight = None
        self.positive_indices = None
        self.negative_indices = None
        self.n_s = None
        self.n_f = None

        self.singular_check = None
        self.contained = None
        self.split = None
        self.weight = None

    def fit(self, X: np.ndarray, y: np.ndarray, wb0: Optional[np.ndarray] = None):
        """
        Returns:
            accuracies, times  (lists across outer iterations)
        """
        self._initial(X, y, wb0)

        accuracies: List[float] = []
        times: List[float] = []
        start_time = time.time()
        break_flag = False

        for ite in range(self.max_ite_num):
            _ = self._project(X, y)

            while np.any(self.singular_check):
                break_flag = self._initial_pinchd(X, y)
                if break_flag:
                    break

            valid_w = self.w[~self.singular_check]
            valid_b = self.b[~self.singular_check]

            criteria = np.matmul(X, valid_w.T) - valid_b
            y_pred = (2 * np.all(criteria >= 0, axis=1) - 1).reshape(-1, 1)
            accuracy = np.mean(y == y_pred)
            accuracies.append(float(accuracy))

            elapsed = time.time() - start_time
            times.append(float(elapsed))

            if not self.silent:
                print(f"Iteration: {ite + 1:3d} | Accuracy: {accuracy:.4f} | Elapsed time: {elapsed:.2f}s")

        self.w = valid_w
        self.b = valid_b
        return accuracies, times

    def _project(self, X: np.ndarray, y: np.ndarray):
        for _ in range(self.num_ite_max):
            grad_w = self._call_weight_th(X, y)

            if np.all(self.singular_check):
                break

            self.w -= self.lr * grad_w
            norm_w = np.linalg.norm(self.w, axis=1, keepdims=True) + 1e-9
            self.w /= norm_w
            self.b = np.min(np.matmul(X[self.positive_indices], self.w.T), axis=0)

        norm_grad = np.linalg.norm(grad_w, axis=1, keepdims=True).squeeze() + 1e-9
        self.singular_check |= (norm_grad < 1e-7) & (np.sum(self.contained, axis=0) < 2 * np.sqrt(self.n_f))
        return None

    def _call_weight_th(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        splits = np.matmul(X, self.w.T) - self.b

        contained = ((splits >= 0).dot(self.adjacency) >= (self.n_splits - 1))

        splits_pos = splits[self.positive_indices]
        if len(splits_pos) < self.n_f:
            threshold_margin = len(splits_pos) - 1
        else:
            threshold_margin = self.n_f - 1

        self.threshold = (np.partition(splits_pos, threshold_margin, axis=0)[threshold_margin] + self.shift_th)

        contained &= (splits <= self.threshold) & (splits >= -self.threshold)

        masked_splits = np.where(contained, splits, 0)
        self.singular_check = np.all(masked_splits >= -1e-9, axis=0)

        self.contained = contained
        self.split = splits

        new_weight0 = self.weight.copy()
        self.weight = np.zeros((X.shape[0], self.n_splits))
        grad_w = np.zeros((self.n_splits, self.n_f))

        for i in range(self.n_splits):
            if self.singular_check[i]:
                self.weight[:, i] = 0
            else:
                self.weight[contained[:, i], i], prob, prob_lr = self.formulate_new_weight(
                    X, y, new_weight0[:, i], contained[:, i], splits[:, i], self.w[i]
                )
                grad_w[i] = self._call_grad_wb_simple(
                    self.weight[contained[:, i], i],
                    X[contained[:, i]],
                    y[contained[:, i]],
                    splits[contained[:, i], i],
                    self.w[i],
                    prob,
                    prob_lr,
                )

        return grad_w

    def formulate_new_weight(
        self,
        X: np.ndarray,
        y: np.ndarray,
        new_weight0: np.ndarray,
        contained: np.ndarray,
        splits: np.ndarray,
        w_prime: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_this_split = X[contained]
        y_this_split = y[contained]
        weight_this_split = new_weight0[contained].reshape(-1, 1)
        splits_this_split = splits[contained].reshape(-1, 1)

        negative_indices = np.where(y_this_split == -1)[0]
        positive_indices = np.where(y_this_split == 1)[0]

        mat_a, inv_ata, prob, prob_lr = self._formulate_mat(y_this_split, splits_this_split)

        alpha = self._renew_weight0(X_this_split, splits_this_split, positive_indices, negative_indices)

        grad_this_split = self._cal_weight_gradient(
            X_this_split, y_this_split, w_prime, alpha, weight_this_split, prob_lr
        )

        weight_this_split = weight_this_split * 1 + grad_this_split * self.lr_grad2

        new_weight = np.maximum(weight_this_split, 0.0)

        new_weight[negative_indices, 0] = self.pva_regression(
            new_weight[negative_indices, 0], splits_this_split[negative_indices, 0]
        )
        new_weight[positive_indices, 0] = self.pva_regression(
            new_weight[positive_indices, 0], -splits_this_split[positive_indices, 0]
        )

        new_weight = self._projection_to_affine_space(new_weight, mat_a, inv_ata)
        return new_weight.squeeze(), prob, prob_lr

    def _renew_weight0(self, X: np.ndarray, splits: np.ndarray, positive_indices: np.ndarray, negative_indices: np.ndarray):
        splits_pos = splits[positive_indices]
        splits_neg = splits[negative_indices]
        positive_idx = np.argmin(splits_pos, axis=0)
        negative_idx = np.argmax(splits_neg, axis=0)
        return X[positive_indices[positive_idx]] - X[negative_indices[negative_idx]]

    def _cal_weight_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_prime: np.ndarray,
        alpha: np.ndarray,
        weight: np.ndarray,
        prob_lr: np.ndarray,
    ) -> np.ndarray:
        w_prime = w_prime.reshape(1, -1)
        gap = w_prime @ alpha.T
        c = prob_lr * y
        gradient = c * X @ (alpha - gap * w_prime).T

        self.w_app = np.sum(weight * c * X, axis=0)
        norm_w_app = np.linalg.norm(self.w_app)
        if norm_w_app == 0:
            self.w_app = alpha * np.max(prob_lr)
            norm_w_app = np.linalg.norm(self.w_app)

        return gradient / np.linalg.norm(self.w_app)

    def pva_regression(self, weight: np.ndarray, splits: np.ndarray) -> np.ndarray:
        order = np.argsort(-splits)
        weight_ordered = weight[order]
        weight_new = self.isotonic_regression(-weight_ordered)
        weight[order] = -weight_new
        return weight

    def isotonic_regression(self, y: np.ndarray) -> np.ndarray:
        n = len(y)
        block_values = []
        block_weights = []
        for i in range(n):
            block_values.append(y[i])
            block_weights.append(1)
            while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
                total_weight = block_weights[-2] + block_weights[-1]
                merged_value = (block_values[-2] * block_weights[-2] + block_values[-1] * block_weights[-1]) / total_weight
                block_values[-2] = merged_value
                block_weights[-2] = total_weight
                block_values.pop()
                block_weights.pop()

        result = np.empty(n, dtype=float)
        idx = 0
        for value, weight in zip(block_values, block_weights):
            result[idx: idx + weight] = value
            idx += weight
        return result

    def _formulate_mat(self, y: np.ndarray, splits: np.ndarray):
        scaled_splits = self.beta * splits
        prob = expit(scaled_splits)
        prob_lr = prob * (1.0 - prob)

        mat_a = np.hstack((y, np.ones_like(y), prob, y * prob_lr))
        mat_ata = mat_a.T @ mat_a
        inv_ata = np.linalg.pinv(mat_ata)
        return mat_a, inv_ata, prob, prob_lr

    def _projection_to_affine_space(self, weight0: np.ndarray, mat_a: np.ndarray, inv_ata: np.ndarray) -> np.ndarray:
        vec_lambda = inv_ata @ (np.array([[0], [2], [1], [0]]) - mat_a.T @ weight0)
        weight = weight0 + mat_a @ vec_lambda
        return weight

    def _call_grad_wb_simple(
        self,
        weight: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        splits: np.ndarray,
        w_prime: np.ndarray,
        prob: np.ndarray,
        prob_lr: np.ndarray,
    ) -> np.ndarray:
        weight = weight.reshape(-1, 1)
        splits = splits.reshape(-1, 1)

        weight_l = weight * prob
        weight_ly = weight_l * y
        B = np.sum(weight_ly, axis=0)

        weight_lr = weight * prob_lr
        weight_lry = weight_lr * y
        Bpx = np.sum(weight_lry * X, axis=0)

        grad_w = 2 * self.beta * B * (w_prime.reshape(1, -1) @ Bpx * w_prime - Bpx)
        return grad_w

    def _initial(self, X: np.ndarray, y: np.ndarray, wb0: Optional[np.ndarray] = None):
        self.n_s, self.n_f = X.shape
        y = y.reshape(-1, 1)

        self.positive_indices = (y.squeeze() == 1)
        self.negative_indices = (y.squeeze() == -1)

        self.singular_check = np.zeros(self.n_splits, dtype=bool)

        self.weight = np.zeros((self.n_s, self.n_splits))
        self.weight0 = np.zeros((self.n_s, self.n_splits))
        self.grad_cum = np.zeros((self.n_s, self.n_splits))
        self.det_weight = np.zeros((self.n_s, self.n_splits))

        if wb0 is not None:
            self.w = wb0[:-1].reshape(1, -1)
        else:
            self.w = 2 * np.random.rand(self.n_splits, self.n_f) - 1

        self.w /= np.linalg.norm(self.w, axis=1, keepdims=True)
        self.b = np.min(np.matmul(X[self.positive_indices], self.w.T), axis=0)

        self.b = np.min(np.matmul(X[self.positive_indices], self.w.T), axis=0)
        if wb0 is None:
            self.singular_check = np.ones(self.n_splits, dtype=bool)
            while np.any(self.singular_check):
                break_flag = self._initial_pinchd(X, y)
                if break_flag:
                    break

    def _initial_pinchd(self, X: np.ndarray, y: np.ndarray):
        splits_all = np.matmul(X, self.w.T) - self.b
        idx_all_positive_side = np.all(splits_all >= 0, axis=1)

        if self.contained is not None:
            idx_aps_contained = idx_all_positive_side & np.any(self.contained, axis=1)
            X_sub = X[idx_aps_contained]
            y_sub = y[idx_aps_contained]
        else:
            X_sub = X[idx_all_positive_side]
            y_sub = y[idx_all_positive_side]

        neg_indices_sub = np.where(y_sub == -1)[0]
        if len(neg_indices_sub) == 0:
            return True

        neg_chosen = np.random.choice(neg_indices_sub)
        neg_point = X_sub[neg_chosen]

        pos_indices_full = np.where(self.positive_indices)[0]
        pos_chosen = np.random.choice(pos_indices_full)
        x_star_new = X[pos_chosen]

        epsilon = 1e-4
        max_iter = 200
        i = 0
        while i < max_iter:
            x_star = x_star_new
            diff_star_neg = x_star - neg_point
            norm_diff = np.linalg.norm(diff_star_neg)

            project_vals = np.matmul(X[pos_indices_full] - neg_point, diff_star_neg.T) / norm_diff
            idx_min = np.argmin(project_vals)
            candidate_diff = x_star - X[pos_indices_full[idx_min]]

            if np.linalg.norm(candidate_diff) < epsilon:
                break

            q_value = np.dot(diff_star_neg, candidate_diff.T) / (np.linalg.norm(candidate_diff) ** 2 + 1e-9)
            q_value = min(1, q_value)

            x_star_new = x_star + q_value * (X[pos_indices_full[idx_min]] - x_star)
            criteria = norm_diff - np.linalg.norm(x_star_new - neg_point)
            if criteria < epsilon:
                break

            i += 1

        w_initial = x_star_new - neg_point
        w_initial /= (np.linalg.norm(w_initial) + 1e-9)
        b_initial = np.min(np.matmul(X[self.positive_indices], w_initial.T))

        sing_indices = np.where(self.singular_check)[0]
        assigned_idx = np.random.choice(sing_indices)
        self.w[assigned_idx] = w_initial
        self.b[assigned_idx] = b_initial
        self.singular_check[assigned_idx] = False
        self.weight0[:, assigned_idx] = 0
        self.grad_cum[:, assigned_idx] = 0

        return False
