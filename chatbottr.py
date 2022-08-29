import nltk
from snowballstemmer import TurkishStemmer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy
import random
import json
from tkinter import *
from functools import partial
from datetime import datetime

root = Tk() 
class BotBubble:

    def __init__(self,master,message="",coorx=400,coory=0):
        self.master = master
        self.frame = Frame(master,bg="#c300ff")
        self.i = self.master.create_window(coorx,coory,window=self.frame)
        Label(self.frame,text=datetime.now().strftime("%Y-%m-%d %H:%M"),font=("Helvetica", 7),bg="#c300ff",fg="#0207a6").grid(row=0,column=0,sticky="w",padx=5)
        Label(self.frame, text=message,font=("Helvetica", 9),bg="#c300ff",fg="#0207a6",anchor=CENTER).grid(row=1, column=0,sticky="w",padx=5,pady=3)
        root.update_idletasks()
        self.master.create_polygon(self.draw_triangle(self.i), fill="#c300ff", outline="light grey")
    def draw_triangle(self,widget):
        x1, y1, x2, y2 = self.master.bbox(widget)
        return x1, y2 - 10, x1 - 15, y2 + 10, x1, y2
bubblesinput = []
bubblesreply = []
with open(r"maindata.json",encoding="utf8") as file:
    data = json.load(file)

stemmer = TurkishStemmer()
words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stemWord(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stemWord(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

'''
def relu_function(x):
    if x<0:
        return 0
    else:
        return x

def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_
'''

model = Sequential()
model.add(Dense(450, input_shape=(len(training[0]), ), activation="relu"))
# Default alpha = 1, Default max_value = None , Default ThreshHold = 0   for relu
model.add(Dense(450, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(184, activation="softmax"))
model.summary()
model.compile(Adam(learning_rate=.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(training, output, epochs=210, verbose=2, batch_size=184)



def gui():

    root.geometry("1800x800")
    root['background']='#9016b5'
    
    chatframe = Canvas(root,bg='#601278', width=1600, height=600)
    chatbotlabel = Label(root,text="Chatbot V1.1",font=("Arial", 60),bg='#9016b5',fg="#0207a6")
    chatbotlabel.place(relx=0.35,rely=0.02)

    
    name_var1=StringVar()
    sentenceentry = Entry(root,textvariable = name_var1,bd=5)
    sentenceentry.config(width=60)
    sentenceentry.bind("<Return>", (lambda event: send(sentenceentry.get())))
    sentenceentry.place(relx=0.35,rely=0.9)
    


    def send(sendedtext):
        turkishchars=["İ","Ğ","Ç","Ş","Ö","Ü","ı","ü","ö","ş","ç","ğ"]
        replacechars=["I","G","C","S","O","U","i","u","o","s","c","g"]
        for temp in range(len(turkishchars)):
            name_var1.set(name_var1.get().replace(turkishchars[temp],replacechars[temp]))
        if bubblesinput:
            chatframe.move(ALL, 0, -60)
        b = BotBubble(chatframe,message=name_var1.get(),coorx=1400,coory=500)
        
        bubblesinput.append(b) 
        def bag_of_words(s, words):
            bag = [0 for _ in range(len(words))]
            s_words = nltk.word_tokenize(s)
            s_words = [stemmer.stemWord(word.lower()) for word in s_words]
                
            for se in s_words:
                for i, w in enumerate(words):
                    if w == se:
                        bag[i] = 1
                
            return numpy.array(bag)
        def chat():  
            inp = name_var1.get()
            
            for temp2 in range(len(turkishchars)):
                inp = inp.replace(turkishchars[temp],replacechars[temp])
                
            if inp.lower() == "kapat":
                root.destroy()
            
            results = model.predict(numpy.asanyarray([bag_of_words(inp, words)]))[0]
            results_index = numpy.argmax(results)
            tag = labels[results_index]
                
            if results[results_index] > 0.70:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                            responses = tg['responses']

                a = BotBubble(chatframe,message=random.choice(responses),coory=500)
                bubblesreply.append(a) 
            else:
                c = BotBubble(chatframe,message="Tam olarak anlayamadım",coory=500)
                bubblesreply.append(c) 

        chatframe.move(ALL,0,-40)
                  
        chat()
        sentenceentry.delete(0,"end")

    sentenceentrybutton = Button(root,text="Gönder",width=5,height=1,command=partial(send,sentenceentry.get()))
    sentenceentrybutton.place(relx=0.60,rely=0.9)
    

    chatframe.place(relx=0.0425,rely=0.2,width=1600,height=600)
    root.resizable(0, 0)
    root.mainloop()

gui()

