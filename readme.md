###第二次任务完成情况

- 逻辑回归
	- 修改分类器、仅对外开放`fit`,`predict`,`score`,`predict_proba`,`get_params`,`set_params`函数
	
- 私有变量和私有方法
	- `_`单下划线仅可以由class内部和其子类访问
	- `__`双下划线仅可以由class本身访问
	- `__xx__`python内部定义的变量名：常用的有:`__doc__`、`__init__`、`__repr__`
	
- 决策树
	- 理解`ID3`,`C4.5`,`CART`的原理、实现步骤;理解预剪枝、后剪枝的原理、实现步骤;
	- 实现`CART`模型;实现`预剪枝`
	- 对Titanic,Wine数据集调参

- svm
	- 理解线性SVM原理(损失函数的导出，求解二次凸优化问题，smo）
	- 了解另一个svm的损失函数：合页损失函数
	- 用梯度下降对荷叶损失函数进行优化
	
- bagging
	- 重写bootstrapping，优化计算时间
	- 用简单的投票实现整合三个分类器的模型（acc变低了？？？）

- boosting
	- 理解adaboost原理
	- 了解adaboost实现步骤
	
- 多分类
	- 了解OVO，OVR原理及实现步骤