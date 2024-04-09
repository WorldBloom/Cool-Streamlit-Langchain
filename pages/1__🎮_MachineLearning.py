import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
# 显示标题
st.title('Titanic :blue[Machine Learning]🚢')
st.markdown("Github开源链接:https://github.com/WorldBloom/Cool-Streamlit-Langchain")
# 显示章节标题
st.header('问题的简单介绍')
#介绍文字
st.markdown("泰坦尼克号Kaggle竞赛是一个挑战，参与者使用机器学习来预测哪些乘客在泰坦尼克号沉船事故中幸存下来。目标是建立一个:blue[预测模型]，通过乘客数据（如姓名、年龄、性别、社会经济阶层等）回答“哪些类型的人更有可能生还”的问题。此竞赛被设计为机器学习初学者的入门挑战。更多细节可以访问[Kaggle上的竞赛概览](https://www.kaggle.com/competitions/titanic/overview)-From ChatGPT")

st.header('数据的观察')

st.markdown("下载好赛题的数据后我们应该观察一下数据大体的形状")
st.markdown("### 训练集的数据")
#导入训练集数据
st.code('''
        import pandas as pd
        df_train = pd.read_csv("titanic_data/train.csv")
        df_train   
        #df_train_head=df_train.head()实际情况可以选择只看前几个数据观察大概情况（默认为前5行）
        ''', language='python')

df_train = pd.read_csv("titanic_data/train.csv")
df_train_head=df_train.head()
st.write(df_train)
st.markdown("可以看到一共有891条数据(没错吧890-0+1)，也就是一共有891个乘客的数据")
st.markdown("一个乘客的数据包含了他的乘客编码，生存状态，仓位等级，姓名，性别等等")
st.markdown("数据集包含以下列：PassengerId、`Survived`（生存状态，这是我们的目标变量）、Pclass、Name、Sex、Age、SibSp、Parch、Ticket、Fare、Cabin和Embarked。")
st.markdown("### 测试集的数据")
#导入测试集的数据
st.code('''
        df_test = pd.read_csv("titanic_data/test.csv")
        df_test   
        #df_test_head=df_test.head()
        ''', language='python')
df_test = pd.read_csv("titanic_data/test.csv")
st.write(df_test)

st.markdown("同样的我们可以观察测试集的形状，测试集相比训练集少了生存状态一列的数据")

#数据处理清洗
st.header('数据处理清洗')
# 检查是否有缺失值
st.markdown('### 缺失值的处理')

st.markdown("让我们的重心回到训练集上来，实际我们可以发现数据中是有非常多的缺失值的（也就是那些显示为None的值），我们需要对这些数据进行处理")

st.markdown("我们首先检查有哪些因子（特征）是有空值的")

#------------------------------------------------------
check_missing_values_train_code = '''
missing_values_train = df_train.isnull().sum()
missing_values_train[missing_values_train > 0])
'''
st.code(check_missing_values_train_code, language='python')
#------------------------------------------------------
missing_values_train = df_train.isnull().sum()
st.write(missing_values_train[missing_values_train > 0])

st.markdown("可以看到age（年龄）有177个空值，Embarked（登船港口？）有2个空值，而`cabin`（船舱）的数据空值高达687！")

st.markdown(
'''
我们决定对这些缺失值做如下的处理(这些都是简单的处理方式)
1. **Age**：使用年龄的中位数填充缺失值。
2. **Cabin**：由于缺失值较多，创建一个新的特征`HasCabin`表示乘客是否有船舱信息（1为有，0为无）:blue[（当然了，大胆点，你可以直接把这一列给删除了😂）]
3. **Embarked**：由于只有2个缺失值，我们将使用最频繁出现的港口填充这些缺失值。（也就是众数）
''')


#------------------------------------------------------
st.code('''
# 填充 'Age' 列中的缺失值为中位数
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)

# 基于 'Cabin' 信息创建 'HasCabin' 特征
df_train['HasCabin'] = df_train['Cabin'].notnull().astype(int)

# 使用最常见的港口信息填充 'Embarked' 列中的缺失值
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
''',language='python')
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)#inplace=True代表将数据真是写入dataframe中
df_train['HasCabin'] = df_train['Cabin'].notnull().astype(int)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
#------------------------------------------------------


st.markdown("好的，现在数据处理完了，我们可以看看处理完长什么样子")
st.code("df_train",language="python")
st.write(df_train)
st.markdown("看上去不错，数据被填充了，也新增加了HasCabin这一列")
st.markdown("我们可以再次使用前面的代码检查一下是否还有空值")
check_missing_values_train_code = '''
missing_values_train = df_train.isnull().sum()
missing_values_train[missing_values_train > 0])
'''
missing_values_train = df_train.isnull().sum()
st.write(missing_values_train[missing_values_train > 0])
st.markdown("可以看到`cabin`还有空值，因为我们只是增加了`HasCabin`的特征，并没有删去原本`cabin`这一列，没关系，我们可以放到后面处理（把他drop掉）")

st.markdown("### 数据编码")
st.markdown("处理完缺失值后整个数据看上去都挺不错了，好像可以直接进行机器学习了，But🤚，观察可以发现我们的数据中是存在`字符型`的数据的，说人话就是文字类型的数据，不是数值型的数据123456789等等")
st.markdown("我们决定对较为友好的特征数据进行编码处理")
st.markdown("较为友好指的是那些种类并不是特别多的特征，比如`Sex`性别，`Embarked`登船港口")
st.markdown("那些不是很友好的数据例如`Name`，`Ticket`等等，种类是在太多了！！，891个人的姓名我总不能编891个号，那样好像没啥意义")
st.code('''
from sklearn.preprocessing import LabelEncoder#这里需要导入额外的库
# 实例化 LabelEncoder
# 对 'Sex' 列进行编码
label_encoder_sex = LabelEncoder()
df_train['Sex'] = label_encoder_sex.fit_transform(df_train['Sex'])
# 对 'Embarked' 列进行编码，先填充缺失值
df_train['Embarked'].fillna('S', inplace=True)#S是最常见的港口
label_encoder_embarked = LabelEncoder()
df_train['Embarked'] = label_encoder_embarked.fit_transform(df_train['Embarked'])
        ''')
# 实例化 LabelEncoder
# 对 'Sex' 列进行编码
label_encoder_sex = LabelEncoder()
df_train['Sex'] = label_encoder_sex.fit_transform(df_train['Sex'])
# 对 'Embarked' 列进行编码，先填充缺失值
df_train['Embarked'].fillna('S', inplace=True)#S是最常见的港口
label_encoder_embarked = LabelEncoder()
df_train['Embarked'] = label_encoder_embarked.fit_transform(df_train['Embarked'])

st.write(df_train)
st.markdown("可以，非常完美，`Sex`被编码成了0 1，`Embarked`被编码成了0 1 2")

st.header("特征工程")
st.markdown("### icing on the cake")

st.markdown('或许你想在进行编码之后我们基本就处理完了数据，我们可以丢掉一些基本没什么意义的特征，马上进行机器学习的处理')
st.markdown('但用你聪明的脑袋瓜想一想，实际上我们可以根据这些没什么意义的特征`创造`出一些有意义的特征，这或许有助于提升机器学习的性能')
st.markdown('让我们回到kaggle赛题页面，在[Data一页](https://www.kaggle.com/competitions/titanic/data)的说明中。所有变量的详细说明都在这里')

# 创建两列布局，第一个参数是每列的相对宽度
col1, col2 = st.columns([2, 1])  # 你可以调整这里的数字来改变列的相对宽度
with col1:  # 使用with语句指定接下来的内容应该放在哪一列
    st.markdown('''
    **变量说明：**

    - **pclass**：社会经济地位（SES）的代理指标
        - 1st = 上层
        - 2nd = 中层
        - 3rd = 下层
    - **age**：年龄，如果小于1岁则为小数。如果年龄是估计的，以xx.5的形式表示
    - **sibsp**：数据集以这种方式定义家庭关系...
        - 兄弟姐妹 = 兄弟、姐妹、继兄、继姐
        - 配偶 = 丈夫、妻子（情妇和未婚夫被忽略）（可以，很炸裂）
        - sibsp其实就是Sibling 和 Spouse 合体，大家可以自己去看原页面
    - **parch**：数据集以这种方式定义家庭关系...
        - 父母 = 母亲、父亲
        - 子女 = 女儿、儿子、继女、继子
        - 有些儿童只是和保姆一起旅行，因此对他们来说parch=0。
    ''')
with col2:
    # 使用empty来创建上方的空白区域
    for _ in range(7):  # 这里的数字10可以根据需要增加或减少
        st.markdown("<br>", unsafe_allow_html=True)
    # 设置图片位置，可能需要调整空白区域的高度以达到居中效果
    st.image('imageandvoice/cat3.png', width=250)
    # 使用empty来创建下方的空白区域



st.markdown('基于SibSp和Parch我们可以FamilySize和IsAlone特征,表示某个乘客的家庭大小 与 是否独身一人')
st.code('''
# 创建 'FamilySize' 特征
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1# 加1是因为还有自己嘛
# 创建 'IsAlone' 特征
df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize'] == 1, 'IsAlone'] = 1
''',language='python')
# 创建 'FamilySize' 特征
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1# 加1是因为还有自己嘛
# 创建 'IsAlone' 特征
df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize'] == 1, 'IsAlone'] = 1

st.markdown("### 删除不必要的列")
st.code("df_train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True #cabin在这里就被删除了",language='python')
# 删除可能对模型不太有用的列
df_train.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)
st.markdown('OK,现在的数据看上去就非常完美了！')
df_train

st.header('机器学习')
st.markdown('在进行完训练集的测试后我们就可以开始进行机器学习了！')
st.markdown(':blue[代码其实很简单！大家重在理解]')

st.code('''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 定义特征和目标变量
X = df_train.drop('Survived', axis=1)  # 从训练数据中除去'Survived'列，剩下的用作特征
y = df_train['Survived']  # 将'Survived'列用作目标变量

# 将数据分割为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = rf_classifier.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print("随机森林模型的准确率是:",accuracy)  # 输出准确率

        ''',language='python')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 定义特征和目标变量
X = df_train.drop('Survived', axis=1)  # 从训练数据中除去'Survived'列，剩下的用作特征
y = df_train['Survived']  # 将'Survived'列用作目标变量

# 将数据分割为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(X_train, y_train)

# 在验证集上进行预测
y_pred = rf_classifier.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)

st.write("### 随机森林模型的准确率是:",accuracy)
st.markdown("随机森林模型在验证集上的准确率为约80.45%!这是一个非常不错的的初始结果!👍👍👍")





st.markdown('''
一些说明（Chat一下，你就知道）

1. **导入必要的库和函数**：代码开始于导入`train_test_split`函数用于数据分割，`RandomForestClassifier`用于构建随机森林模型，以及`accuracy_score`用于计算预测准确率。

2. **定义特征和目标变量**：从`df_train`数据帧中分离出特征（`X`）和目标变量（`y`）。特征包括了除了`Survived`列之外的所有列，而`Survived`列是作为目标变量。

3. **分割数据为训练集和验证集**：使用`train_test_split`函数将数据随机分割为训练集和验证集，其中验证集占总数据的20%。

4. **初始化随机森林分类器**：创建一个随机森林分类器实例，指定树的数量为100，并设置随机状态以确保结果的可重复性。

5. **训练模型**：使用训练数据（特征和目标变量）训练随机森林分类器。

6. **在验证集上进行预测**：使用训练好的模型在验证集的特征上进行预测。

7. **计算准确率**：通过比较验证集的真实目标变量和模型预测结果来计算准确率。
            ''')



##其他模型的选择与可视化
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

# 假设df_train是已经加载进来的Pandas DataFrame
# 这里用随机数据生成一个示例DataFrame
# 你应该用实际的数据替换这一部分
# Streamlit应用开始
st.markdown('### 不同模型的预测效果')
st.markdown('当然了，你完全可以选择不同的模型进行训练预测，一切依据你的喜好与认知，这里:blue[只提供不同的测试选项]，具体代码相信大家可以自己写出来')
st.markdown('当然了模型基本都是sklearn的模型，xgboost需要额外下包')

# 选择模型类型
model_type = st.selectbox(
    '选择一个模型类型📝',
    ('随机森林', '逻辑回归', '支持向量机','XGBoost','决策树','K最近邻','朴素贝叶斯','梯度提升树','AdaBoost')
)
# 根据选定的模型类型显示参数调整UI
if model_type == '随机森林':
    n_estimators = st.slider('树的数量', min_value=10, max_value=200, value=100)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
elif model_type == '逻辑回归':
    C = st.slider('正则化强度的倒数', min_value=0.01, max_value=1.0, value=1.0)
    model = LogisticRegression(C=C, random_state=42)
elif model_type == '支持向量机':
    C = st.slider('正则化参数', min_value=0.01, max_value=1.0, value=1.0)
    model = SVC(C=C, probability=True, random_state=42)
elif model_type == 'XGBoost':
    max_depth = st.slider('最大树深度', min_value=3, max_value=10, value=6)
    n_estimators = st.slider('树的数量', min_value=10, max_value=200, value=100)
    learning_rate = st.slider('学习率', min_value=0.01, max_value=0.3, value=0.1)
    model = xgb.XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
elif model_type == '决策树':
    max_depth = st.slider('最大树深度', min_value=1, max_value=10, value=3)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
elif model_type == 'K最近邻':
    n_neighbors = st.slider('邻居数量', min_value=1, max_value=10, value=5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
elif model_type == '朴素贝叶斯':
    model = GaussianNB()
elif model_type == '梯度提升树':
    n_estimators = st.slider('树的数量', min_value=10, max_value=200, value=100)
    learning_rate = st.slider('学习率', min_value=0.01, max_value=0.3, value=0.1)
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
elif model_type == 'AdaBoost':
    n_estimators = st.slider('树的数量', min_value=10, max_value=200, value=50)
    learning_rate = st.slider('学习率', min_value=0.01, max_value=1.0, value=1.0)
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

# 分割数据
X = df_train.drop('Survived', axis=1)  # 从训练数据中除去'Survived'列，剩下的用作特征
y = df_train['Survived']  # 将'Survived'列用作目标变量
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_val)

# 计算准确率
accuracy = accuracy_score(y_val, y_pred)

# 显示准确率
st.write(f'### 模型的准确率高达: `{accuracy}`')

st.header('真正的预测')
st.markdown('前面的机器学习都是在对训练集进行操作，现在我们需要将训练好的模型应用到测试集上')
st.markdown('为了保证数据的格式等都是对应的，我们需要对测试集进行与训练集同样的处理')

st.markdown('### 对测试集数据进行处理')

st.code('''
# 加载测试数据集
df_test = pd.read_csv('titanic_data/test.csv')

# 使用训练数据集的中位数填充 'Age' 和 'Fare' 列的缺失值
df_test['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Fare'].fillna(df_train['Fare'].median(), inplace=True)

# 基于 'Cabin' 信息创建 'HasCabin' 特征
df_test['HasCabin'] = df_test['Cabin'].notnull().astype(int)

# 创建 'FamilySize' 特征
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

# 创建 'IsAlone' 特征
df_test['IsAlone'] = 0
df_test.loc[df_test['FamilySize'] == 1, 'IsAlone'] = 1

# 编码处理
df_test['Sex'] = label_encoder_sex.transform(df_test['Sex'])
df_test['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Embarked'] = label_encoder_embarked.transform(df_test['Embarked'])
        
# 删除可能对模型没什么用的列
df_test.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

        ''')

# 加载测试数据集
df_test = pd.read_csv('titanic_data/test.csv')
# 使用训练数据集的中位数填充 'Age' 和 'Fare' 列的缺失值
df_test['Age'].fillna(df_train['Age'].median(), inplace=True)
df_test['Fare'].fillna(df_train['Fare'].median(), inplace=True)

# 基于 'Cabin' 信息创建 'HasCabin' 特征
df_test['HasCabin'] = df_test['Cabin'].notnull().astype(int)

# 创建 'FamilySize' 特征
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

# 创建 'IsAlone' 特征
df_test['IsAlone'] = 0
df_test.loc[df_test['FamilySize'] == 1, 'IsAlone'] = 1

# 使用训练集中最常见的港口信息填充 'Embarked' 列的缺失值
df_test['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Sex'] = label_encoder_sex.transform(df_test['Sex'])
df_test['Embarked'] = label_encoder_embarked.transform(df_test['Embarked'])
# 删除可能对模型较少帮助的列
df_test.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# 展示测试数据集的前几行以验证
df_test

st.markdown("### 使用随机森林模型对测试集进行预测")
st.code('''
# 使用随机森林模型对测试集进行预测
y_pred = rf_classifier.predict(df_test)
        
# 输出预测结果
st.write(y_pred)
        
# 将预测结果保存到CSV文件中，保存一个包含乘客ID和他们的生存预测的文件
# 创建一个DataFrame来保存结果
submission = pd.DataFrame({
    "PassengerId": test_df_with_ids['PassengerId'],
    "Survived": y_pred
})
        
# 保存结果到CSV文件
submission.to_csv('titanic_data/titanic_predictions.csv', index=False)

        ''')
# 使用随机森林模型对测试集进行预测
y_pred = rf_classifier.predict(df_test)

# 如果你需要将预测结果保存到CSV文件中，假设你想要保存一个包含乘客ID和他们的生存预测的文件
# 首先，重新加载测试数据集以获取PassengerId
test_df_with_ids = pd.read_csv('titanic_data/test.csv')
# 确保预测结果的数量与乘客ID的数量相匹配
assert len(y_pred) == len(test_df_with_ids), "结果的数量与乘客ID的数量不匹配"

# 创建一个DataFrame来保存结果
submission = pd.DataFrame({
    "PassengerId": test_df_with_ids['PassengerId'],
    "Survived": y_pred
})
# 保存结果到CSV文件
submission.to_csv('titanic_data/titanic_predictions.csv', index=False)
st.header("预测的结果")
# 显示保存的文件头部以确认
st.write(submission)
st.markdown("现在我们就可以预测出这些人的生存状态了，我们可以把结果交到kaggle上看看结果如何😆")
st.markdown("在kaggle赛题页面右上角点集Submit Prediction即可提交结果")

st.image("imageandvoice/score.png",width=1200)
st.markdown("很棒！得分是0.75119，这样的得分在滚动排行榜上的排名大约在1w名左右，但是没有过1w，能否超越1w就看大家努力了😏")

st.header("来点拓扑")
st.markdown("敬请期待...")