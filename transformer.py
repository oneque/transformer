# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 08:26:37 2020

@author: ilyoung
"""

import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf


########################################################
#Position Encoding
#########################################################




# dmodel  = 512 : input/output의 차원수
#num_layers  = 6 : encoder/decoder의 stack layer수
# num_heads  = 8 : attentin시 병렬진행하는 갯수 dmodel/num_heads = head 차원수
# dff  = 204 : ff시의 hidden layer의 크기(차원수 )

# 기존 seq2seq와 다른 점은 encoder/decoder내애 n개의 encoders/decoders로 구성
# RNN은 사용하지 않는다, sequence가 한번에 입력

# Positional Encoding : RNN을 사용하지 않기에 position에 대한 정보 필요
# 각 word에 해당 position에 대한 정보 재공 필요
# embeding vector + position encoding을 한단위로 encoder input을 사용

# position 방법 :  PE(pos, 2i)=sin(pos/10000^(2i/dmodel))     -->  i가 짝수 일때 
#                  PE(pos, 2i+1)=cos(pos/10000^(2i/dmodel))   -->  i가 홀수 일때
# p : 입력문장내 embeding vector의 위치(y축 개념), i는 embeding vector내 차원의 index(x축 개념)

#              dmodec(512 demension)
#        i                           i(가로방향)
#       am             *  
#        a
#  student    
#        pos(세로방향) 

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__() # 부모 class method를 사용하고자 할때
                                               # 
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    # amgle을 구하는 함수 ,tf.cast(d_model, tf.float32) cast(var, tf.float32) var을 float32로 변환
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
        # tf.range :Creates a sequence of numbers.
        # tf.range(limit, delta=1, dtype=None, name='range')
        # tf.range(start, limit, delta=1, dtype=None, name='range')
        #Returns 1-D Tensor of type dtype.

    # 배열의 짝수 인덱스(2i)에는 사인 함수 적용
    sines = tf.math.sin(angle_rads[:, 0::2])

    # 배열의 홀수 인덱스(2i+1)에는 코사인 함수 적용
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    print(pos_encoding.shape)
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[3], :]

#sample_pos_encoding = PositionalEncoding(50, 512)

# plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
# plt.xlabel('Depth')
# plt.xlim((0, 512))
# plt.ylabel('Position')
# plt.colorbar()
# plt.show()

########################################################
#Attention
#########################################################
#Encoder Self-Attention : encoder에서 각각에 대한 attention
#Masked Decoder Self-Encode : decoder에서 사용
#Encode-Decorder Attention : Decoder에서 사용
#self attention은 Query=Key=Value가 동일 
#Encodr-Decoder Attention에서는 Query= decoder vector, key=value= encoder vector
# 여기서 vector가 같다는 것은 값이 같다라는 것이 아니라 출처가 같다라는 의미



########################################################
#Encoder 구조
#########################################################
# n개의 (Multi-Head Attetion + Position-wise FFNN)의 Stack
#output은 dmodel과 동일



########################################################
# Self Attention in Encder
#########################################################
# 일반적 Attention 개념 
#주어진 Query에 대하여 Key값과의 유사도를 구하고 이를 value에 반영
#유사도가 모두 반영된 Value값을 가중합하여 return
# Q = Query : t 시점의 디코더 셀에서의 은닉 상태
# K = Keys : 모든 시점의 인코더 셀의 은닉 상태들
# V = Values : 모든 시점의 인코더 셀의 은닉 상태들
# 결국 전체시전에 대하여 정리하면 
#Q = Querys : 모든 시점의 디코더 셀에서의 은닉 상태들
#K = Keys : 모든 시점의 인코더 셀의 은닉 상태들
#V = Values : 모든 시점의 인코더 셀의 은닉 상태들

#But self attentiond은 Query, Key,Value vector의 출처가 동알
# In Trasformer 
# Q : 입력 문장의 모든 단어 벡터들
# K : 입력 문장의 모든 단어 벡터들
# V : 입력 문장의 모든 단어 벡터들

##################################################
# Q, K, V 얻기
#################################################
# dmodel의 차원을 가지는 words의 vectors를 가지고 Q,K,V를 얻는 것이 아니다.
# dmodel의 차원을 가지는 word마다 Q,K,V를 얻는 과정을 거친다.
# 512 차원을 8ea(multihead)의 64차원으로 Q,K,V로 분활 한다.

##### 가중치 행렬  
# (1,512) weight matrix 크기는 dmodelX(dmodel/num_heads) : 512 X 64 크기

##### Q,K,V 행열
# a= 3 -1 2   b= 3 4      앞의 행의 갯수가 뒤의 열의 갯수와 같아야 한다. 
#    4 0 5       6 -2     알의 열의 갯수가 뒤의 행의 행의 갯수와 같아야 한다.
#                4  3
#    (2,3)       (3,2)      
# a X c sixe= 2,2         결과의 행은 앞의 행, 열은 뒤의 열
 
# (1,64) 결과 위해서는 앞의 행은 (1,x) 뒤의 열은 (x,64)
# 따라서 size (1,512) X (512, 64)으로 연산되어야 한다.
## 중요한 것은 각 word마다 Q.K,V에 대하여 정리하여야 한다.
#  한 문장을 이후는 I am a student 각각에 대하여 Q,K,V 값이 정리 되어 있어야 한다.

####Scaled dot-product Attention
# Q,K,V가 정리되었기 
#기존 Attention의 score fuction은 Q.K의 내적
# transformer에서는 normmalize by K vetor 차원의 제곱근으로 나누어주는 scaled dot.Product attention 적용
# Q별 K와 유사고 확인 
#  for q in Q : 
#     for k in K 
#         Attention_score.append(scaled( q x k))
#         softmax(Attention_score)
#         soft_max x V : 가중치 추가된 value
#   loop마다 해당 word의 가중치가 추가된 value가 update 된다.
# 실제 적용은 문장 행렬에 가중치 vector을 적용하여 한번에 계산
# Q x KT(전치) = (64,64)
# softmax(Q x KT(전치)/sqrt(dkdemension)) X V = attention Matrix 로 간단히 계산 가능
# attention Matrix의 크기는 (seqence_length, dv)

#####################################################
#스케일드 닷-프로덕트 어텐션 구현하기
######################################################

def scaled_dot_product_attention(query, key, value, mask):
  # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
  # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
  # value 크기 : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
  # padding_mask : (batch_size, 1, 1, key의 문장 길이)

  # Q와 K의 곱. 어텐션 스코어 행렬.
  matmul_qk = tf.matmul(query, key, transpose_b=True)
  #print("matmul_qk : ",matmul_qk)


  # 스케일링
  # dk의 루트값으로 나눠준다.
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  #print("depth:  ",depth)
  
  logits = matmul_qk / tf.math.sqrt(depth)
  #print("logits:  ",logits)
  # 마스킹. 어텐션 스코어 행렬의 마스킹 할 위치에 매우 작은 음수값을 넣는다.
  # 매우 작은 값이므로 소프트맥스 함수를 지나면 행렬의 해당 위치의 값은 0이 된다.
  if mask is not None:
    logits += (mask * -1e9)

  # 소프트맥스 함수는 마지막 차원인 key의 문장 길이 방향으로 수행된다.
  # attention weight : (batch_size, num_heads, query의 문장 길이, key의 문장 길이)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  # output : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
  output = tf.matmul(attention_weights, value)

  return output, attention_weights

# 임의의 Query, Key, Value인 Q, K, V 행렬 생성
# np.set_printoptions(suppress=True)
# temp_k = tf.constant([[10,0,0],
#                       [0,10,0],
#                       [0,0,10],
#                       [0,0,10]], dtype=tf.float32)  # (4, 3)

# temp_v = tf.constant([[   1,0],
#                       [  10,0],
#                       [ 100,5],
#                       [1000,6]], dtype=tf.float32)  # (4, 2)
# temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3) 

# 함수 실행
# temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
# print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
# print(temp_out) # 어텐션 값

# temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)
# temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
# print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
# print(temp_out) # 어텐션 값

# temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)  # (3, 3)
# temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
# print(temp_attn) # 어텐션 분포(어텐션 가중치의 나열)
# print(temp_out) # 어텐션 값
   
####
#Multi-head Attention
####
#여러번의 어텐션을 병렬로 사용하는 것이 더 효과적
#병합된 Multi-head Attention의 크기는 (seq_length, dmodel)
#결론적은 input matrix와 size가 같다.

####
#Multi-head Attention 구현하기
####
# 필요 weight matrix 1) weight matrix(WQ,WK,WV), attention head merger후 output weight와 
# 연결되는 WO의 2가지로 연결됨.
# weight matrix 곱하는 것은 구현상에서 입력을 닐집층을 지나게 함으로서 구현된다.
# Multi head attention 구현 단계
# 1) WQ,WK,WV에 해당되는 dmodel 크기의 밀집층을 지나게 한다.
# 2) 지정된 head수 만큼 나눈다(split)
# 3) 나누었던 head들을 연결한다.
# 4) WO에 해당되는 밀집층을 지나게 한다.
#############################################
class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    # d_model을 num_heads로 나눈 값.
    # 논문 기준 : 64
    self.depth = d_model // self.num_heads

    # WQ, WK, WV에 해당하는 밀집층 정의
    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    # WO에 해당하는 밀집층 정의
    self.dense = tf.keras.layers.Dense(units=d_model)

  # num_heads 개수만큼 q, k, v를 split하는 함수
  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # 1. WQ, WK, WV에 해당하는 밀집층 지나기
    # q : (batch_size, query의 문장 길이, d_model)
    # k : (batch_size, key의 문장 길이, d_model)
    # v : (batch_size, value의 문장 길이, d_model)
    # 참고) 인코더(k, v)-디코더(q) 어텐션에서는 query 길이와 key, value의 길이는 다를 수 있다.
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # 2. 헤드 나누기
    # q : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # k : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # v : (batch_size, num_heads, value의 문장 길이, d_model/num_heads)
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # 3. 스케일드 닷 프로덕트 어텐션. 앞서 구현한 함수 사용.
    # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
    # (batch_size, query의 문장 길이, num_heads, d_model/num_heads)
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # 4. 헤드 연결(concatenate)하기
    # (batch_size, query의 문장 길이, d_model)
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # 5. WO에 해당하는 밀집층 지나기
    # (batch_size, query의 문장 길이, d_model)
    outputs = self.dense(concat_attention)

    return outputs

####
#Padding Mask
####
#<PAD> token이 있는 경우에  아주 직은 값을 인가하여 softmax후 0이 되도록  유도
##################

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, key의 문장 길이)
  return mask[:, tf.newaxis, tf.newaxis, :]

#print(create_padding_mask(tf.constant([[1, 21, 777, 0, 0]])))

#위 벡터를 스케일드 닷 프로덕트 어텐션의 인자로 전달하면, 스케일드 닷 프로덕트 어텐션에서는 위 벡터에다가
# 매우 작은 음수값인 -1e9를 곱하고, 이를 행렬에 더해주어 해당 열을 전부 마스킹하게 되는 것입니다.

####################################################################
##Position-wise FFNN
####################################################################
#Position-wise FFNN은 Encode/decoder 공통, FFNN(Fully-connected FFNN)
# FFNN(x)=MAX(0,xW1+b1)W2+b2
# x -> F1=xW1+b1 --> 화성화함수 :ReLu, F2 = max(0,F1) --> F3=F2W2+b2
#W1=(dmodel, dff)m W2=(dff,dmodel) 크기 유지
# W1,W2, b1,b2는 같은 층에서는 반드시 동일해애 함.
##############
#outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
#outputs = tf.keras.layers.Dense(units=d_model)(outputs)


########################################################################
#1) Residual connection(잔치연결) 
#########################################################################
#### Residual connection
# H(x)=x+F(x) : x가 input으로 사용된 함수의 output에 x를 add하여 출력하는 것.
# 입력과 출력이 동일한 차원을 유지해야 한다.
# 결론적으로 multi-head attention input + multi-head attention output의 결과
########################################################################
#2) Layer Normalization(층 정규화)
#########################################################################
# Residual connection 입력을 x라 하고  Layer Normalization까지 완료한 Layer를 
# LN=LayerNorm(x+Sublayer(x))라고 표현 가능 
# tensor의 마지막 차원(dmodel)에 대하여 평균과 분산을 구하고 어떤 수식을 통해 정규화 
# 각 층별로 정규화 수행후 xi는 lni vector로 정규화 된다.
# lni=LayerNorm(xi)
# 정규화 수식 1) 평균과 분산을 통한 정규화 2) 감마와 베타를 도입
# 1)  평균과 분산을 통한 정규화 : x'(i,k)= [(xi,k)−μi]/SQRT(σ^2i+ϵ) ϵ: 0방지 목적   
# 2)  γ=[1,1,1,,,1] 와 β=[0,0,,,,,0]
#  lni=γx'i + β =LayerNorm(xi)
# γ와 β는 학습 가능한 파라미터
# 케라스에서는 LayerNormalization()를 이미 제공하고 있으므로, 이를 가져와 사용

#########################################################################
##### encoder 구현하기
##########################################################################

def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

  # 인코더는 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 멀티-헤드 어텐션 (첫번째 서브층 / 셀프 어텐션)
  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': padding_mask # 패딩 마스크 사용
      })

  # 드롭아웃 + 잔차 연결과 층 정규화
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  # 포지션 와이즈 피드 포워드 신경망 (두번째 서브층)
  outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 드롭아웃 + 잔차 연결과 층 정규화
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

#########################################################################
########
#########################################################################

def encoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")

  # 인코더는 패딩 마스크 사용
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  # 포지셔널 인코딩 + 드롭아웃
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  # 인코더를 num_layers개 쌓기
  for i in range(num_layers):
    outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

#########################################################################
####  12. From Encoder To Decoder
##########################################################################
# #nea encoder layer를 진행후 최종 output이 Decoder의 input으(multi-head Attention)로 사용
# #nea의 decoder layer가 진행되는데 각 layer 진행시 마다 encoder의 최종 출력이 input으로 사용


#########################################################################
####  13. Masked Multi-head self Attention(Decoder 1st sub layer)
##########################################################################
#
# encoder와 동일하게 embedding vector와 positinal ending을 input을 사용
# encoder의 multi head self atterntion과 다르게 masked multi-head self attent 적용
# mask적용 사유 : 예상 번역울 input으로 받아 다음 word를 예측하는 것이 목표인데 
# 학습방법상 teaching Learing 방식으로 답안도 동시에 제공되는 사유로 맞추어야할 next word를 
# 감추어야 할 필요가 있음(Look ahead mask)
# encoder와 동일하게 attention score 정리, 미래단어는 참고하지 못하도록 masking
#
# Masking의 종류 
# 코더의 셀프 어텐션 : 패딩 마스크를 전달
#디코더의 첫번째 서브층인 마스크드 셀프 어텐션 : 룩-어헤드 마스크를 전달 <-- 지금 배우고 있음.
#디코더의 두번째 서브층인 인코더-디코더 어텐션 : 패딩 마스크를 전달
# masking 하고자 하는 곳ㅇ는 1, masking 하지 않는 곳에는 0을 return
# padding Masking도 동시 구현 

# 디코더의 첫번째 서브층(sublayer)에서 미래 토큰을 Mask하는 함수
def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1] # input 문장의 구조는 (seqlength, dmodel) [1]은 1축을 의미(가로방향)
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  #0축의 맨 마지막부터 0보다 큰 것은 0으로 변경
  #band_part(tf.ones((seq_len, seq_len)), -1, 0)
  
  padding_mask = create_padding_mask(x) # 패딩 마스크도 포함
  return tf.maximum(look_ahead_mask, padding_mask)


###############################################################################
#### Decorder 2nd sub layer(Multi Head attention)
###############################################################################
#인코더의 첫번째 서브층 : Query = Key = Value (self)
#디코더의 첫번째 서브층 : Query = Key = Value (slef)
#디코더의 두번째 서브층 : Query : 디코더 행렬 / Key = Value : 인코더 행렬
#
###############################################################################
#Decoder layer
###############################################################################
#

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

  # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  # 멀티-헤드 어텐션 (첫번째 서브층 / 마스크드 셀프 어텐션)
  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs, 'key': inputs, 'value': inputs, # Q = K = V
          'mask': look_ahead_mask # 룩어헤드 마스크
      })

  # 잔차 연결과 층 정규화
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  # 멀티-헤드 어텐션 (두번째 서브층 / 디코더-인코더 어텐션)
  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1, 'key': enc_outputs, 'value': enc_outputs, # Q != K = V
          'mask': padding_mask # 패딩 마스크
      })

  # 드롭아웃 + 잔차 연결과 층 정규화
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  # 포지션 와이즈 피드 포워드 신경망 (세번째 서브층)
  outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)

  # 드롭아웃 + 잔차 연결과 층 정규화
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)

##############################################################################
### stack decoder layers
###############################################################################
#
def decoder(vocab_size, num_layers, dff,
            d_model, num_heads, dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')

  # 디코더는 룩어헤드 마스크(첫번째 서브층)와 패딩 마스크(두번째 서브층) 둘 다 사용.
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  # 포지셔널 인코딩 + 드롭아웃
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  # 디코더를 num_layers개 쌓기
  for i in range(num_layers):
    outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads,
        dropout=dropout, name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)


##############################################################################
#### Transformer 구현하기
###############################################################################
#

def transformer(vocab_size, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):

  # 인코더의 입력
  inputs = tf.keras.Input(shape=(None,), name="inputs")

  # 디코더의 입력
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  # 인코더의 패딩 마스크
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  # 디코더의 룩어헤드 마스크(첫번째 서브층)
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask, output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)

  # 디코더의 패딩 마스크(두번째 서브층)
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
  enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[inputs, enc_padding_mask]) # 인코더의 입력은 입력 문장과 패딩 마스크

  # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
  dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff,
      d_model=d_model, num_heads=num_heads, dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  # 다음 단어 예측을 위한 출력층
  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

###############################################################################
#### Transformer Parameter
###############################################################################
#
# small_transformer = transformer(
#     vocab_size = 9000,
#     num_layers = 4,
#     dff = 512,
#     d_model = 128,
#     num_heads = 4,
#     dropout = 0.3,
#     name="small_transformer")

# tf.keras.utils.plot_model(
#     small_transformer, to_file='small_transformer.png', show_shapes=True)

##############################################################################
#### Loss fuction 정의
##############################################################################
#
# MAX_LENGTH=1000 # 임시 가정
# def loss_function(y_true, y_pred):
#     y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

#     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
#     mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
#     loss = tf.multiply(loss, mask)

#     return tf.reduce_mean(loss) 


##############################################################################
#### Learning rate
##############################################################################
# 랜스포머의 경우 학습률(learning rate)은 고정된 값을 유지하는 것이 아니라, 학습 경과에
# 따라 변하도록 설계하였습니다. 아래의 공식으로 학습률을 계산하여 사용하였으며, 
# warmup_steps의 값으로는 4,000을 사용
# lrate=dmodel^(−0.5) ×min(step_num^−0.5, step_num × warmup_steps^(−1.5))
#

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

##############################################################################

# sample_learning_rate = CustomSchedule(d_model=128)

# plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
