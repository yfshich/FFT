import net_load_data
# net_load_data.load_data()
train_data,validation_data,test_data = net_load_data.data_transform()
 
import net_network as net
net1 = net.Network([784,30,10])
min_batch_size = 10
eta = 3.0
epoches = 30
net1.SGD(train_data,min_batch_size,epoches,eta,test_data)
print "complete"
