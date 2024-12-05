import pubnub
import json
from pubnub.pnconfiguration import PNConfiguration
from pubnub.pubnub          import PubNub, SubscribeListener

class MyListener(SubscribeListener):
    def message( self, pubnub, data ):
        fw=open("data.txt","w")
        data=data.message
        print(data)
        fw.write(str(data))
        fw.close()       
        
        

pnconfig = PNConfiguration()
print("Hello")
pnconfig.subscribe_key = 'sub-c-0fe02006-4dfd-11e8-9c03-eafb26272eb6'
pnconfig.publish_key = 'pub-c-3a67b87b-a0cf-4a60-8394-d2b78b30a7ff'

pubnub = PubNub(pnconfig)

pubnub.add_listener(MyListener())
pubnub.subscribe().channels("fogr").execute()