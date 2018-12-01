## titanic 一周预测报告

### 主要思路：
	1. 特征工程：
		PassengerID,Name采用直接扔掉的方式（有关Name的处理后续有空可以跟进。
		Cabin采用根据有无记录分为0,1的方式（考虑到没有登记的应该比较危险）（后续有空可以将其分为ABCD和缺失）
		Age采用平均值填充（从Name预测后续有空可以写）
		Pclass,Sex,Embarked只有分类意义，故将其升维
		数据归一化采用均值方差归一化
		
	2. 预测： 
		采用LogisticRegression
		有空可以尝试为某些特征加入权重或多项式项（有没有办法通过可视化数据判断如何加入权重或多项式项？
	
	3. 评估标准：
		在metrics模块中内置四个评估标准：acc_score,precision,recall,F1score
		用matplotlib绘制roc曲线
		直接计算auc
		
	4. 数据划分方法：
		在train_test_split模块中内置两种数据划分函数（返回一组X_train等）
		另外内置交叉验证函数（直接返回得到的对应分数的均值）