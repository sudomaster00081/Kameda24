# Install Transformers
!pip install transformers==3
# To get model summary
!pip install torchinfo

import numpy as np
import pandas as pd
import re
import torch
import random
import torch.nn as nn
import transformers
import matplotlib.pyplot as plt
import torch

# specify GPU
device = torch.device("cuda")
url = "https://raw.githubusercontent.com/sudomaster00081/Temporary/main/chatbotintent.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(url)

df.head()
df
# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Check class distribution
df['label'].value_counts(normalize=True)
# In this example, we have used all the utterances for training purposes
train_text, train_labels = df['text'], df['label']
from transformers import DistilBertTokenizer, DistilBertModel

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

text = ["this is a distil bert model.", "data is oil"]
# Encode the text
encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
print(encoded_input)

# get length of all the messages in the train set
seq_len = [len(i.split()) for i in train_text]
pd.Series(seq_len).hist(bins=10)
# Based on the histogram we are selecting the max len as 8
max_seq_len = 8

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length = max_seq_len,
    # pad_to_max_length=True,
    padding='max_length',
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Define a batch size
batch_size = 16

# Wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# Sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# DataLoader for the train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()
        # dense layer
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 13)
        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        cls_hs = self.bert(sent_id, attention_mask=mask)[0][:, 0]

        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc3(x)

        # apply softmax activation
        x = self.softmax(x)
        return x
# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
    param.requires_grad = False

model = BERT_Arch(bert)

# push the model to GPU
model = model.to(device)

from torchinfo import summary
summary(model)

from transformers import AdamW
# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

from sklearn.utils.class_weight import compute_class_weight
# compute the class weights
class_wts = compute_class_weight(class_weight="balanced", classes= np.unique(train_labels), y= train_labels)
print(class_wts)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy = nn.NLLLoss(weight=weights)

# empty lists to store training and validation loss of each epoch
train_losses=[]
# number of training epochs
epochs = 200
# We can also use learning rate scheduler to achieve better results
# lr_sch = torch.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# function to train the model
def train():

  model.train()
  total_loss = 0

  # empty list to save model predictions
  total_preds = []

  # iterate over batches
  for step, batch in enumerate(train_dataloader):

    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>13,}  of  {:>13,}.'.format(step, len(train_dataloader)))
    # push the batch to GPU
    batch = [r.to(device) for r in batch]
    sent_id, mask, labels = batch
    # get model predictions for the current batch
    preds = model(sent_id, mask)
    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)
    # add on to the total loss
    total_loss = total_loss + loss.item()
    # backward pass to calculate the gradients
    loss.backward()
    # clip the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # update parameters
    optimizer.step()
    # clear calculated gradients
    optimizer.zero_grad()

    # We are not using a learning rate scheduler as of now
    # lr_sch.step()
    # model predictions are stored on GPU. So, push it to CPU
    preds = preds.detach().cpu().numpy()
    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)

  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in the form of (number of samples, no. of classes)
  total_preds = np.concatenate(total_preds, axis=0)
  # returns the loss and predictions
  return avg_loss, total_preds

for epoch in range(epochs):

    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    #train model
    train_loss, _ = train()
    print(f'\nTraining Loss: {train_loss:.3f}')
    # append training and validation loss
    train_losses.append(train_loss)
    # it can make your experiment reproducible, similar to set random seed to all options where there needs a random seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f'\nTraining Loss: {train_loss:.3f}')

data = {
    "intents": [
        {"tag": "greeting",
         "responses": ["Hello", "How are you doing?", "Greetings!", "How do you do?", "I'm here to assist you"]},
        {"tag": "age",
         "responses": ["I am 25 years old", "I was born in 1998", "My birthday is July 3rd and I was born in 1998", "03/07/1998"]},
        {"tag": "date",
         "responses": ["I am available all week", "I don't have any plans", "I am not busy"]},
        {"tag": "name",
         "responses": ["My name is Testbot", "I'm Testbot", "You can call me Testbot"]},  # Corrected the response
        {"tag": "goodbye",
         "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]},
        {"tag": "appointment_booking",
         "responses": ["Sure, let's schedule that for you.", "I can help you with appointment booking.",
                        "Absolutely, I'll assist you with scheduling.", "Let's find a suitable time for your appointment.",
                        "Booking an appointment is no problem.", "Certainly, we can set up an appointment.",
                        "I'm here to help you book your appointment.", "Booking a consultation is easy with me.",
                        "I'll guide you through the appointment booking process.",
                        "Feel free to let me know your preferred time for the appointment."]},
        {"tag": "medical_inquiry",
         "responses": ["I'm not a doctor, but I'll do my best to help.", "It's essential to consult a healthcare professional for accurate advice.",
                        "Consider reaching out to a medical professional for personalized guidance.",
                        "I recommend discussing this with your doctor for precise information.",
                        "While I can provide general information, consulting a healthcare professional is crucial.",
                        "It's important to seek advice from a qualified medical professional regarding your health concerns.",
                        "I'm here to offer support, but please consult with a medical professional for specific health queries.",
                        "For accurate information, it's best to consult with a healthcare professional.",
                        "I'm not a substitute for professional medical advice; consider consulting a doctor for personalized guidance.",
                        "Your health is important. It's advisable to consult with a healthcare professional for personalized assistance."]},
        {"tag": "test_results_inquiry",
         "responses": ["I'll check on your test results for you.", "Allow me a moment to retrieve your test results.",
                        "I'll get the information on your test results.", "Let me find your test results for you.",
                        "I'm here to assist with your test results.", "I'll look into your test results shortly.",
                        "I'll provide the details about your test results.", "Just a moment, I'll fetch your test results.",
                        "I'm checking your test results now.", "I'll get back to you with your test results."]},
        {"tag": "hospital_facilities",
         "responses": ["Our hospital provides a range of facilities to meet your needs.", "Feel free to explore our various hospital amenities.",
                        "We have state-of-the-art facilities to ensure your comfort.",
                        "Discover the comprehensive facilities we offer at our hospital.",
                        "From modern equipment to dedicated staff, our facilities are designed to serve you.",
                        "Explore the diverse facilities available in our hospital.",
                        "We aim to provide top-notch facilities for your healthcare journey.",
                        "Our hospital is equipped with a variety of amenities for your convenience.",
                        "Discover the range of services and facilities available at our hospital.",
                        "We strive to make your experience comfortable with our excellent facilities."]},
        {"tag": "health_advice",
         "responses": ["Maintaining a balanced diet is crucial for good health.", "Regular exercise is essential for a healthy lifestyle.",
                        "Adequate sleep is important for overall well-being.", "Stay hydrated throughout the day for optimal health.",
                        "Consult with a healthcare professional for personalized health advice.",
                        "Managing stress is key to maintaining good health.", "Preventive care plays a significant role in long-term well-being.",
                        "Incorporate fruits and vegetables into your daily diet.", "Physical activity is essential for cardiovascular health.",
                        "Listen to your body and seek medical advice if needed."]},
        {"tag": "billing_and_payments",
         "responses": ["For billing inquiries, please contact our billing department.",
                        "You can find information about payments on our website.",
                        "Feel free to inquire about your medical bills at our billing desk.",
                        "We're here to help with any questions regarding payments.",
                        "For billing assistance, visit our customer service center.",
                        "Information about payments can be found on our official website.",
                        "Feel free to ask about billing-related concerns at the front desk.",
                        "We're available to assist you with any billing or payment inquiries.",
                        "You can get details about payments by calling our customer support.",
                        "Feel free to inquire about payment options and billing details."]},
        {"tag": "hospital_location",
         "responses": ["The hospital is located at [Address].", "You can find us at [Address].",
                        "Our hospital is situated at [Address].", "The address of the hospital is [Address].",
                        "You'll find our hospital at [Address].", "Our location is [Address].",
                        "We are located at [Address].", "You can visit us at [Address].",
                        "The hospital's address is [Address].", "Our location can be found at [Address]."]},
        {"tag": "other",
         "responses": ["I'll do my best to help you.", "Feel free to ask anything.", "I'm here to assist you.",
                        "Let me know if you have more questions.", "Ask away!", "I'm here for any inquiries.",
                        "Feel free to inquire about anything.", "I'm at your service.", "Ask me anything!",
                        "I'm here to provide information."]}
    ]
}


def get_prediction(str):
 str = re.sub(r'[^a-zA-Z ]+', '', str)
 test_text = [str]
 model.eval()

 tokens_test_data = tokenizer(
 test_text,
 max_length = max_seq_len,
#  pad_to_max_length=True,
 padding='max_length',
 truncation=True,
 return_token_type_ids=False
 )
 test_seq = torch.tensor(tokens_test_data['input_ids'])
 test_mask = torch.tensor(tokens_test_data['attention_mask'])

 preds = None
 with torch.no_grad():
   preds = model(test_seq.to(device), test_mask.to(device))
 preds = preds.detach().cpu().numpy()
 preds = np.argmax(preds, axis = 1)
 print('Intent Identified: ', le.inverse_transform(preds)[0])
 return le.inverse_transform(preds)[0]

def get_response(message):
  intent = get_prediction(message)
  for i in data['intents']:
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  print(f"Response : {result}")
  # return "Intent: "+ intent + '\n' + "Response: " + result
  return(intent)

get_response('hello there')

tgh = 0
while tgh<5:
    message = input(">>")
    tempval = get_response(message)
    tgh = tgh+1
    
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Save the entire model
bert.save_pretrained("distilbert_chatbot_model")

# Optionally, save the tokenizer as well
tokenizer.save_pretrained("distilbert_chatbot_model")   


