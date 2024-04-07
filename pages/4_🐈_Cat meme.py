import streamlit as st
import base64

# 加载图片
folder_path = 'imageandvoice/'

image2_path = folder_path + 'cat2.png'

image_width = 400
# 创建三列用于放置图片
st.image(image2_path, caption='爆笑猫.jpg',width=image_width)

# 按钮用于触发声音
if st.button('点击让小猫嘲笑你'):
    # 读取音频文件
    audio_file_path = folder_path + 'catlaugh.mp3'
    audio_file = open(audio_file_path, 'rb')
    audio_bytes = audio_file.read()
    
    # 将音频文件转换为Base64编码
    b64 = base64.b64encode(audio_bytes).decode()
    audio_html = f'<audio src="data:audio/mp3;base64,{b64}" controls autoplay>'
    
    # 使用HTML显示音频播放器
    st.components.v1.html(audio_html, height=50)
