#%%
# Import the Gtts module for text  
# to speech conversion 
try:
    from gtts import gTTS 
    import os
    print("----Libraries Loaded----")
except:
    print("----Libraries Not Loaded----")
#%% 

textfile = open("sample.txt", "r")
myText = textfile.read().replace("\n", " ")

#%%
# Language we want to use 
language = 'en'

output = gTTS(text=myText, lang=language, slow=False)

output.save("output.mp3")
fh.close()
#%%
# Play the converted file 
os.system("start output.mp3")
#%%