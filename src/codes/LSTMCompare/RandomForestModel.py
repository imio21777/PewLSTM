"""
Random Forest Regression Model
使用sklearn的RandomForestRegressor进行停车行为预测
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import pickle
import os


class RandomForestModel:
    """
    随机森林回归模型
    用于预测停车场的到达/离开车辆数
    """
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            random_state: 随机种子
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def prepare_data(self, x, y):
        """
        准备训练数据
        Args:
            x: [num_samples, seq_len, features] - 3D array
            y: [num_samples] - 1D array
        Returns:
            x_flat: [num_samples, seq_len * features] - 2D array
            y: [num_samples]
        """
        # 将3D数据展平为2D (Random Forest需要2D输入)
        batch_size, seq_len, features = x.shape
        x_flat = x.reshape(batch_size, seq_len * features)
        
        return x_flat, y
    
    def train(self, train_x, train_y):
        """
        训练模型
        Args:
            train_x: [batch_size, 24, 5] - 训练输入
            train_y: [batch_size * 24] - 训练标签
        """
        # 准备数据
        train_x_flat, train_y = self.prepare_data(train_x, train_y.reshape(len(train_x), -1))
        
        # 归一化
        train_x_scaled = self.scaler_x.fit_transform(train_x_flat)
        train_y_scaled = self.scaler_y.fit_transform(train_y)
        
        # 训练
        # 训练
        # Reshape y to [samples, 24] for multi-output regression
        train_y_reshaped = train_y_scaled.reshape(len(train_x_scaled), -1)
        self.model.fit(train_x_scaled, train_y_reshaped)
    
    def predict(self, test_x):
        """
        预测
        Args:
            test_x: [batch_size, 24, 5] - 测试输入
        Returns:
            predictions: [batch_size * 24] - 预测结果
        """
        # 准备数据
        test_x_flat, _ = self.prepare_data(test_x, np.zeros((len(test_x), 24)))
        
        # 归一化
        test_x_scaled = self.scaler_x.transform(test_x_flat)
        
        # 预测
        pred_scaled = self.model.predict(test_x_scaled)
        
        # 反归一化
        pred = self.scaler_y.inverse_transform(pred_scaled)
        
        return pred.ravel()
    
    def save(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_x': self.scaler_x,
                'scaler_y': self.scaler_y
            }, f)
    
    def load(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler_x = data['scaler_x']
            self.scaler_y = data['scaler_y']
