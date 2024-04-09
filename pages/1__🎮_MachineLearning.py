import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
# 显示标题
st.title('Titanic :blue[Machine Learning]🚢')
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
label_encoder = LabelEncoder()
# 对 'Sex' 和 'Embarked' 这两个分类变量进行编码
df_train['Sex'] = label_encoder.fit_transform(df_train['Sex'])
# 为了保证 'Embarked' 使用同一个 label_encoder 实例，这里先填充缺失值，再进行转换
df_train['Embarked'] = label_encoder.fit_transform(df_train['Embarked'].fillna('S'))
        ''')
# 实例化 LabelEncoder
label_encoder = LabelEncoder()
# 对 'Sex' 和 'Embarked' 这两个分类变量进行编码
df_train['Sex'] = label_encoder.fit_transform(df_train['Sex'])
# 为了保证 'Embarked' 使用同一个 label_encoder 实例，这里先填充缺失值，再进行转换
df_train['Embarked'] = label_encoder.fit_transform(df_train['Embarked'].fillna('S'))
st.write(df_train)
st.markdown("可以，看上去很完美，`Sex`被编码成了0 1，`Embarked`被编码成了0 1 2")








