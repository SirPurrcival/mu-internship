import pickle

with open("prepspikes_pre", 'rb') as f:
   pre = pickle.load( f)
   
with open("prepspikes_post", 'rb') as f:
   post = pickle.load( f)
      
with open("join_results", 'rb') as f:
   join = pickle.load( f)