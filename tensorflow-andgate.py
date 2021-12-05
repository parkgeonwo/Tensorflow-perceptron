#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 텐써플로우 2.0을 이용해서 AND 퍼셉트론 구현하기


# In[1]:


import  tensorflow  as   tf

tf.random.set_seed(777)  # 시드를 설정한다. # 일치된 값으로 생성되게 하기 위해서 777로 설정합니다.
import  numpy  as  np      
from  tensorflow.keras.models  import  Sequential   # 신경망 모델 구성
from  tensorflow.keras.layers  import  Dense  # 신경망 층 / 완전 연결계층 
from  tensorflow.keras.optimizers  import   SGD  # 경사감소법 ( 오차가 가장 작은 지점으로 경사하강 )
from  tensorflow.keras.losses   import   mse    #  오차함수 

# 데이터 준비
x = np.array( [ [0, 0], [1, 0], [0, 1], [1, 1] ] )
y = np.array( [ [0], [0], [0], [1] ] )          # AND 게이트 데이터 준비

#모델 구성하기 

model = Sequential()

#단층 퍼셉트론 신경망 만들기

model.add( Dense( 1, input_shape =( 2,  ), activation ='linear')  ) 

# 모델 준비하기

model.compile( optimizer= SGD(),         # 경사하강법 종류
                     loss= mse,                        # 오차함수 지정
                     metrics = ['acc'] )      # list 형태로 평가지표를 전달한다.  

# 학습 시키기 

model.fit(x, y, epochs = 300) 


# In[ ]:





# In[2]:


# 입력데이터를 모델에 넣어서 어떤값으로 예상하는지 확인하기
result = model.predict(x)
print(result)     


# In[ ]:





# In[3]:


result = model.predict(x)
print(result.round())    


# In[ ]:





# In[4]:


print( list ( model.get_weights() ) )           # 최종 가중치 출력


# In[ ]:





# In[5]:


print( model.evaluate(x,y) )             # 오차와 정확도 출력


# In[ ]:





# In[ ]:




