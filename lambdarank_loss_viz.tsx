import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

const LambdaRankVisualization = () => {
  const [si, setSi] = useState(0.5);
  const [sj, setSj] = useState(-0.5);
  
  // 计算损失值
  const calculateLoss = (si, sj) => {
    return Math.log(1 + Math.exp(-(si - sj)));
  };
  
  // 计算梯度
  const calculateGradient = (si, sj) => {
    return -1 / (1 + Math.exp(si - sj));
  };
  
  // 生成损失曲线数据（固定 sj，改变 si）
  const generateLossData = () => {
    const data = [];
    for (let x = -3; x <= 3; x += 0.1) {
      const loss = calculateLoss(x, sj);
      data.push({
        si: parseFloat(x.toFixed(2)),
        loss: parseFloat(loss.toFixed(4))
      });
    }
    return data;
  };
  
  // 生成梯度曲线数据
  const generateGradientData = () => {
    const data = [];
    for (let diff = -4; diff <= 4; diff += 0.1) {
      const gradient = calculateGradient(diff + sj, sj);
      data.push({
        diff: parseFloat(diff.toFixed(2)),
        gradient: parseFloat(gradient.toFixed(4))
      });
    }
    return data;
  };
  
  const lossData = generateLossData();
  const gradientData = generateGradientData();
  const currentLoss = calculateLoss(si, sj);
  const currentGradient = calculateGradient(si, sj);
  const diff = si - sj;

  return (
    <div className="w-full max-w-5xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl shadow-lg">
      <h2 className="text-3xl font-bold text-gray-800 mb-2 text-center">
        LambdaRank Pairwise Loss 可视化
      </h2>
      <p className="text-center text-gray-600 mb-6">
        损失函数：L<sub>ij</sub> = log(1 + exp(-(s<sub>i</sub> - s<sub>j</sub>)))
      </p>
      
      {/* 控制面板 */}
      <div className="bg-white rounded-lg p-6 mb-6 shadow-md">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-semibold text-green-700 mb-2">
              正样本分数 s<sub>i</sub>: {si.toFixed(2)}
            </label>
            <input
              type="range"
              min="-3"
              max="3"
              step="0.1"
              value={si}
              onChange={(e) => setSi(parseFloat(e.target.value))}
              className="w-full h-2 bg-green-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
          
          <div>
            <label className="block text-sm font-semibold text-red-700 mb-2">
              负样本分数 s<sub>j</sub>: {sj.toFixed(2)}
            </label>
            <input
              type="range"
              min="-3"
              max="3"
              step="0.1"
              value={sj}
              onChange={(e) => setSj(parseFloat(e.target.value))}
              className="w-full h-2 bg-red-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4 mt-6 text-center">
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-xs text-gray-600">分数差</div>
            <div className="text-xl font-bold text-blue-700">
              s<sub>i</sub> - s<sub>j</sub> = {diff.toFixed(2)}
            </div>
          </div>
          <div className="bg-orange-50 p-3 rounded-lg">
            <div className="text-xs text-gray-600">当前损失</div>
            <div className="text-xl font-bold text-orange-700">
              L = {currentLoss.toFixed(3)}
            </div>
          </div>
          <div className="bg-purple-50 p-3 rounded-lg">
            <div className="text-xs text-gray-600">梯度 ∂L/∂s<sub>i</sub></div>
            <div className="text-xl font-bold text-purple-700">
              {currentGradient.toFixed(3)}
            </div>
          </div>
        </div>
      </div>

      {/* 损失曲线 */}
      <div className="bg-white rounded-lg p-6 mb-6 shadow-md">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          损失函数随 s<sub>i</sub> 变化（s<sub>j</sub> = {sj.toFixed(2)}）
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={lossData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="si" 
              label={{ value: 'si (正样本分数)', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: '损失 L', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <ReferenceLine x={si} stroke="#ef4444" strokeDasharray="5 5" label="当前 si" />
            <ReferenceLine x={sj} stroke="#10b981" strokeDasharray="5 5" label="sj" />
            <Line 
              type="monotone" 
              dataKey="loss" 
              stroke="#f59e0b" 
              strokeWidth={3}
              dot={false}
              name="损失 L"
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-sm text-gray-600 mt-2 text-center">
          💡 当 s<sub>i</sub> &gt; s<sub>j</sub> 时，损失快速下降；当 s<sub>i</sub> &lt; s<sub>j</sub> 时，损失很大
        </p>
      </div>

      {/* 梯度曲线 */}
      <div className="bg-white rounded-lg p-6 mb-6 shadow-md">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          梯度随分数差 (s<sub>i</sub> - s<sub>j</sub>) 变化
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={gradientData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="diff" 
              label={{ value: 'si - sj (分数差)', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: '梯度 ∂L/∂si', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip />
            <ReferenceLine x={0} stroke="#888" strokeDasharray="3 3" />
            <ReferenceLine y={0} stroke="#888" strokeDasharray="3 3" />
            <ReferenceLine x={diff} stroke="#ef4444" strokeDasharray="5 5" label="当前差值" />
            <Line 
              type="monotone" 
              dataKey="gradient" 
              stroke="#8b5cf6" 
              strokeWidth={3}
              dot={false}
              name="梯度"
            />
          </LineChart>
        </ResponsiveContainer>
        <p className="text-sm text-gray-600 mt-2 text-center">
          💡 梯度是负的 Sigmoid 函数：当差值很负时梯度接近 -1，差值很正时梯度接近 0
        </p>
      </div>

      {/* 解释说明 */}
      <div className="space-y-3">
        <div className="bg-gradient-to-r from-green-50 to-blue-50 p-4 rounded-lg shadow">
          <h3 className="font-semibold text-green-900 mb-2">🎯 核心思想</h3>
          <ul className="text-sm text-gray-700 space-y-1 ml-4">
            <li>• <strong>目标</strong>：让正样本 i 的分数高于负样本 j</li>
            <li>• <strong>损失设计</strong>：当 s<sub>i</sub> &gt; s<sub>j</sub> 时损失小，s<sub>i</sub> &lt; s<sub>j</sub> 时损失大</li>
            <li>• <strong>平滑性</strong>：使用 log-exp 形式，处处可导</li>
          </ul>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 p-4 rounded-lg shadow">
          <h3 className="font-semibold text-purple-900 mb-2">📐 梯度特性</h3>
          <ul className="text-sm text-gray-700 space-y-1 ml-4">
            <li>• <strong>s<sub>i</sub> ≪ s<sub>j</sub></strong>：梯度 ≈ -1，强烈推动 s<sub>i</sub> 上升</li>
            <li>• <strong>s<sub>i</sub> ≈ s<sub>j</sub></strong>：梯度 ≈ -0.5，中等强度调整</li>
            <li>• <strong>s<sub>i</sub> ≫ s<sub>j</sub></strong>：梯度 ≈ 0，几乎不调整（已经正确排序）</li>
          </ul>
        </div>

        <div className="bg-gradient-to-r from-orange-50 to-yellow-50 p-4 rounded-lg shadow">
          <h3 className="font-semibold text-orange-900 mb-2">✨ 为什么这样设计？</h3>
          <p className="text-sm text-gray-700 leading-relaxed">
            与 step 函数类似，我们希望 "s<sub>i</sub> &gt; s<sub>j</sub> 就好，否则就惩罚"。
            但 step 不可导，所以用 <strong>sigmoid 的变体</strong>（即这个 log-exp 形式）来平滑逼近。
            这样既保留了 "排序正确时损失小" 的性质，又提供了平滑的梯度用于优化。
          </p>
        </div>
      </div>
    </div>
  );
};

export default LambdaRankVisualization;