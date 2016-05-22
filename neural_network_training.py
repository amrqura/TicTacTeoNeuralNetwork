'''
Written By: Amr Koura.

In this script I will show how to train the neural network to play simple 
game like tic-tac-toe. the main algorithm for selecting the best move
can be found in tic-tac-toe wikipedia page.
https://en.wikipedia.org/wiki/Tic-tac-toe#Strategy

the idea is simple , the network will read all possible positions for tic-tac-toe 
game and then will examine the best move for that.

our neural  network will have the following structure.

input layer: 18 nodes.
hidden layer:36 nodes
output layer: 9 nodes

the 18 postion will show the game from the player point of view , where the first half 9 postions
will contains zero if the player didn't play this square and will contains one if he plays the corresponding
square. similarly , the next half will show the opponents player position , so the square will filled with one
if the opponent play this square , otherwise will be zero 

we are going to use tensorflow library , the output from this script should be 
tensorflow check point file , that can be read later by the game itself to start
playing

'''

import tensorflow as tf
import numpy as np
import random
from numpy import shape


def get_empty_side(board_status,player):
    '''
    try to get an empty side "middle square" from the board.
    parameter:
        the game board in the current situation
        the player who have the turn to play
    Output:
        the cell number of the empty side, otherwise return -1  
    '''
    xs,ys=np.where(board_status==-1)
    for i in range(len(xs)):
        if is_side(xs[i],ys[i]):
            return (xs[i]*3)+ys[i]
    return -1    
def get_empty_corner(board_status,player):
    '''
    try to get an empty corner from the board.
    parameter:
        the game board in the current situation
        the player who have the turn to play
    Output:
        the cell number of the empty corner, otherwise return -1  
    '''
    xs,ys=np.where(board_status==-1)
    for i in range(len(xs)):
        if is_corner(xs[i], ys[i]):
            return (3*xs[i])+ys[i]
    return -1    
def try_opposite_side(board_status,player):
    '''
    get the opponents position and try to find which corner is filled by the opponent, and 
    then fill it by the player move
    parameter:
        the game board in the current situation
        the player who have the turn to play
    Output:
        the cell number that will is opposite to opponent corner, otherwise return -1    
    '''
    opponent=get_opponent(player)
    xs,ys=np.where(board_status==opponent)
    for i in range(len(xs)):
        if is_corner(xs[i], ys[i]):
            x,y=get_opposite_corner(xs[i], ys[i])
            if(board_status[x][y]==-1):
                return (x*3)+y

    return -1        
def get_opposite_corner(x,y):
    '''
    get the opposite corner from the given corner.
    parameter :
        x,y position.
    output:
        x,y of opposite corner
    '''
    if x==0 and y==0:
        return 2,2
    if x==0 and y==2:
        return 2,0
    if x==2 and y==0:
        return 0,2
    if x==2 and y==2:
        return 0,0
    return -1
def block_fork(board_status,player):
    '''
    this function will try to block an opportunity where the opponent can win in two different ways
    Parameter:
        the game board in the current situation
        the player who have the turn to play
    Output:
        the cell number that will block the opponent from winning , otherwise return -1
    
    '''
    opponent=get_opponent(player)
    return  try_to_fork(board_status, opponent)

def is_side(x,y):
    '''
    simple function to check if the element in x,y is side or not
    parameter:
        x axis
        y axis
    return:
        True if element in (x,y) is side , otherwise: false
    '''
    if x==1 and y==0:
        return True
    if x==1 and y==2:
        return True
    if x==0 and y==1:
        return True
    if x==2 and y==1:
        return True
    return False
def is_corner(x,y):
    '''
    simple function to check if the element in x,y is corner or not
    parameter:
        x axis
        y axis
    return:
        True if element in (x,y) is corner , otherwise: false
    '''
    if x==0 and y==0:
        return True
    if x==0 and y==2:
        return True
    if x==2 and y==0:
        return True
    if x==2 and y==2:
        return True
    return False
def valid_fork(arr,player):
    '''
    check if the array with the player is enough to create a fork or not, basicly the array should have two free spots with one spot with player
    parameter:
        array from three elements.
    return:
        if the array can make valid fork , return 1 , otherwise return 0
    '''
    if np.count_nonzero(arr!=player) ==2 and np.count_nonzero(arr!=-1) ==1:
        return 1
    else:
        return 0
    
def try_to_fork(board_status,player):
    '''
    this function will try to create an opportunity where the player can win in two different ways
    Parameter:
        the game board in the current situation
        the player who have the turn to play
    Output:
        the cell number that will create fork for the current player , otherwise return -1
    
    '''
    #first get the free places
    
    xs,ys=np.where(board_status==-1)
    # try all the free places
    for i in range(len(xs)):
        sum=0    
        if is_corner(xs[i], ys[i]):
            #check for corner, here we should check 3 , row,coloumn , [diagonal or diagonal transpose]
            if xs[i]==ys[i]: # this is in case , (0,0) and (2,2), will take diagonal 
                sum=valid_fork(board_status[xs[i]], player)+valid_fork(board_status[:,ys[i]], player)+valid_fork(np.diag(board_status), player)
            else: # this is case (0,2),(2,0) will take row,coloumn, diagonal transpose
                sum=valid_fork(board_status[xs[i]], player)+valid_fork(board_status[:,ys[i]], player)+valid_fork(np.diag(np.rot90(board_status)), player)       
        else:
            #check the  row and colomn
            sum=valid_fork(board_status[xs[i]], player)+valid_fork(board_status[:,ys[i]], player)
        if sum>=2:
            # if the sum =2 , this mean that this place will create fork
            return (xs[i]*3)+ys[i]             
    
    
    return -1


def try_to_block(board_status,player):
    '''
    this function will try to find the move that will block the opponent to win.
    Parameter:
        the game board in the current situation
        the player who have the turn to play
    Output:
        the cell number that will block the opponent from winning , otherwise return -1
    '''
    opponent=get_opponent(player)
    
    return try_to_win(board_status, opponent)

def try_to_win(board_status,player):
    '''
    this function will try to find the move that will make the player win.
    Parameter:
        the game board in the current situation
        the player who have the turn to play
    Output:
        the cell number that will make the player winning , otherwise return -1
    '''
    #get number of move that this player has played before
    previous_moves=np.where(board_status==player)[0]
    # if the previous moves less than 2 , this means that the player can't win now
    if previous_moves.size<2:
        return -1
    # check rows
    for i in range(3):
        tmp_arr=board_status[i]
        # make sure that in the row , two elements equals to the player and there is one empty space
        if np.count_nonzero(tmp_arr==player) ==2 and np.count_nonzero(tmp_arr==-1) ==1:  
            return (i*3)+np.where(tmp_arr==-1)[0][0] # [0][0] because I want to get the first element in the tuple and the first element pf the array 
    
    # check colomns
    for i in range(3):
        tmp_arr=board_status[:,i]
        if np.count_nonzero(tmp_arr==player) ==2 and np.count_nonzero(tmp_arr==-1) ==1:  
            return i+3*(np.where(tmp_arr==-1)[0][0]) # [0][0] because I want to get the first element in the tuple and the first element of the array 
    
    
    #check diagonal
    tmp_arr=np.diag(board_status)
    if np.count_nonzero(tmp_arr==player) ==2 and np.count_nonzero(tmp_arr==-1) ==1:
        x=np.where(tmp_arr==-1)[0][0] # get the element row number
        return (3*x)+x
    
    #check diagonal transponse
    tmp_arr=np.diag(np.rot90(board_status))
    if np.count_nonzero(tmp_arr==player) ==2 and np.count_nonzero(tmp_arr==-1) ==1:
        x=np.where(tmp_arr==-1)[0][0]
        return (3*x)+(2-x)
    return -1

def get_best_move(board_status,player):
    '''
    this function implements the main algorithm for selecting the best moves for player given specific board_status
    input: the board_status represent the current board , will be array[9]
    player: name of the player who has the turn to play
    output: will be number in range 0 -> 8 : that will select the best move for player:player in this current station: board_status
    '''
    board_status=np.array(board_status)
    board_status=board_status.reshape(3,3)
    
    # first is try to win
    out=try_to_win(board_status, player)
    if out!=-1:
        return out
    # second: try to block the opponent from winning
    out=try_to_block(board_status, player)
    if out!=-1:
        return out
    #third : try to create fork for the current player
    out=try_to_fork(board_status, player)
    if out!=-1:
        return out
    
    out=block_fork(board_status, player)
    if out!=-1:
        return out
    #check if the center is avaliable , then take the center
    if board_status[1][1]==-1:
        return 4
    #try to get the opposite corner:
    out=try_opposite_side(board_status,player)
    if out!=-1:
        return out
    
    #get empty corner
    out=get_empty_corner(board_status, player)
    if out!=-1:
        return out
    
    out=get_empty_side(board_status, player)
    if out!=-1:
        return out
    
    return 3



def encode_input(game_board,player):
    '''
    this function will encode the input.
    
    parameter: array of 9 elements represents the current board, and the player that has the turn to play
    output: array of 18 represents the player view of the board , the first half will represents players point ogf view
    where later half , 9 will represents the opponents
    
    for game_board:
    -1: mean that slot is empty
    1: mean that the current player takes this slot
    0: mean that the opponents takes this slot
    '''
    output=np.array([-1]*18) # since -1 mean that the place is empty
    game_board=np.array(game_board)
    # see what I have played
    indicies=np.where(game_board==player)[0] # since where return tuple and we are interested by the the first item
    if indicies.size>0:
        output[indicies]=1
        # set all 
        for index in indicies:
            output[index+9]=0
    #get the opponent 
    opponent=get_opponent(player)
    indicies=np.where(game_board==opponent)[0]
    if indicies.size>0:
        output[indicies]=0
        for index in indicies:
            output[index+9]=1
    return output    

def get_opponent(input):
    if input==0:
        return 1
    else:
        return 0

def encode_output(num):
    '''
    this function recieve the input as number and will return array of 9
    where the position corresponding to num will equals one
    '''
    output=np.zeros(9)
    output[num]=1
    return output


def get_training_data_set():
    '''
    this function should return the training data set that will be feeded to the neural network.
    the ouput: will be tuple having two items:
    1- training_data : should be matrix, where shape=[#correct possible position, 18]
    2- training_labels: should be matrix, where shape=[#correct possible position, 9]
    '''
    print 'start'
    data = [line for line in open("allboards.txt", 'r')]
    x_moves=np.zeros(len(data))
    y_moves=np.zeros(len(data))
    # maximum nuber of rows is len(data)*2, will be shortered later
    trainig_data_set=np.zeros(shape=(len(data)*2,18))
    training_labels=np.zeros(shape=(len(data)*2,9))
    row=0
    for i in range(len(data)):
        if i==100:
            print "100"
        board_state=data[i]
        #convert strind to array
        # 1: represents x plays
        # 0: represent y plays
        # empty: represent no one play
        board_game=list(board_state.rstrip('\n')) # .rstrip('\n') to remove the new line
        #convert it to array
        #board_game=np.array(len(board_status))
        #for j in range(len(board_status)):
         #   board_game[j]=board_status[j]
        #board_game=[n for n in board_status] 
        # examine who should play
        board_game=[x.strip() for x in board_game]
        board_game=[-1 if (x=='')  else x for x in board_game] # convert the empty space to -1
        board_game=[1 if (x=='1')  else x for x in board_game]
        board_game=[0 if (x=='0')  else x for x in board_game]
        if board_game.count(1)<board_game.count(0):
            #in case X plays less than O , then X turns
            encoded_row=encode_input(board_game, 1)
            trainig_data_set[row]=encoded_row

            best_move=get_best_move(board_game,1)
            x_moves[i]=best_move
            training_labels[row]=encode_output(best_move)
            row=row+1
        elif board_game.count(1)>board_game.count(0):
            #in case O plays less than X , then O turns
            encoded_row=encode_input(board_game, 0)
            trainig_data_set[row]=encoded_row
            
            
            best_move=get_best_move(board_game,0)
            training_labels[row]=encode_output(best_move)
            row=row+1

            y_moves[i]=best_move
        else:
            #in case both plays same number then , X or O can play
            encoded_row=encode_input(board_game, 1)
            trainig_data_set[row]=encoded_row
            
            
            # in case that the both has same plays , then any one of them cab play
            best_move=get_best_move(board_game,1)
            x_moves[i]=best_move
            training_labels[row]=encode_output(best_move)
            row=row+1
            
            encoded_row=encode_input(board_game, 0)
            trainig_data_set[row]=encoded_row
            
            best_move=get_best_move(board_game,0)
            y_moves[i]=best_move
            training_labels[row]=encode_output(best_move)
            row=row+1           
    return trainig_data_set[0:row],training_labels[0:row]
    

def move_still_possible(S):
    return not (S[S==-1].size == 0)

def move_at_random(S, p):
    xs, ys = np.where(S==-1)

    i = np.random.permutation(np.arange(xs.size))[0]
    
    S[xs[i],ys[i]] = p

    return S


def move_was_winning_move(S, p):
    #if np.max((np.sum(S, axis=0)) * p) == 3:
    for i in range(3):
        if np.max(S[i])==np.min(S[i])==p:
            return True
        
        if np.max(S[:,i])==np.min(S[:,i])==p:
            return True

    #if np.max((np.sum(S, axis=1)) * p) == 3:
    #if np.max(S[2])==np.min(S, axis=1)==p:
     #   return True

    #if (np.sum(np.diag(S)) * p) == 3:
    if np.max(np.diag(S))==np.min(np.diag(S))==p:
        return True

    #if (np.sum(np.diag(np.rot90(S))) * p) == 3:
    if np.max(np.diag(np.rot90(S)))==np.min(np.diag(np.rot90(S)))==p:
        return True

    return False

symbols = {1:'x', -1:' ', 0:'o'}

def print_game_state(S):
    B = np.copy(S).astype(object)
    for n in [-1, 0, 1]:
        B[B==n] = symbols[n]
    print B
    
counts=np.zeros(9)

def add_noise(gredient_tensors,t,previous_noise,sess):
    '''
    This function is used to add noise to the gredient.
    Accroding to "Neural network for pattern recognition" book
    Adding noise to the gredient is successful way to 
    deal with the overfitting problem.
    
    we will not add fixed noise , but we are going to add
    time-dependent gaussian noise to the gredient g at each time t:
    According to paper:
    http://arxiv.org/pdf/1511.06807v1.pdf
    
    g_t=g_t+N(0,vraiance)
    
    where we are going to decay the variance 
    
    variance=n/((1+t)^gama)
    n selected from  {0.01, 0.3, 1.0} and gama=0.55
    
    Parameter:
        gredient tensors: list of training gredients
        t: the training step
        previous_noise: noise from previous step
        sess: the tensorflow session
    return:
        assign_ops: list of operation that will update the gredient
        noise: the new computed noise
    '''
    import random
    print 'previous noise value is '+str(previous_noise)
    n=[0.01, 0.3, 1.0]
    gamma=0.55
    variance=n[t%3] / ((1+t)**gamma)
    noise=previous_noise+np.float32(random.uniform(-1*variance, variance))
    assign_ops=[]
    #r=np.float32(random.uniform(-0.03, 0.03))
    for gredient_tensor in gredient_tensors:
        arr=sess.run(gredient_tensor)
        assign_op = gredient_tensor.assign(arr+noise)
        assign_ops.append(assign_op)
    #training_data=training_data+noise
    return assign_ops,noise       

def main():
    training_data,trainig_labels=get_training_data_set()
    training_data1=training_data
    trainig_labels1=trainig_labels
    training_data2=training_data+0.0001
    trainig_labels2=trainig_labels
    training_data3=training_data+0.0002
    trainig_labels3=trainig_labels
    training_data4=training_data+0.0003
    trainig_labels4=trainig_labels
    training_data=np.zeros(shape=(len(training_data1)*4,18))
    trainig_labels=np.zeros(shape=(len(training_data1)*4,9))
    
    # fill them
    training_data[0:len(training_data1)]=training_data1
    trainig_labels[0:len(training_data1)]=trainig_labels1
    
    training_data[len(training_data1):2*len(training_data1)]=training_data2
    trainig_labels[len(training_data1):2*len(training_data1)]=trainig_labels2
    
    training_data[2*len(training_data1):3*len(training_data1)]=training_data3
    trainig_labels[2*len(training_data1):3*len(training_data1)]=trainig_labels3
    
    training_data[3*len(training_data1):4*len(training_data1)]=training_data4
    trainig_labels[3*len(training_data1):4*len(training_data1)]=trainig_labels4
    
    
    #start training the network
    import tensorflow as tf
    import random
    #place holder represent the input and output from the neural network
    X=tf.placeholder(tf.float32,shape=(None,18))
    y_=tf.placeholder(tf.float32,shape=(None,9))
    # W1 represent the weghts from input layer to hidden layer
    # shouls be matrix of size [18 * 36] since the input layer contains
    # 18 nodes and the hidden layer contains 36
    W1 =tf.Variable(tf.random_uniform([18, 36], -1.0, 1.0))
    #bias for hidden layers, should contains 36 nodes
    b1 = tf.Variable(tf.zeros([36]))
    #tensor that computes hidden layer
    hidden=tf.nn.relu(tf.matmul(X,W1)+b1)
    
    # from hidden layer to output layer
    #W2: (36,9) , b2=(9)
    W2 =tf.Variable(tf.random_uniform([36, 9], -1.0, 1.0))
    b2 = tf.Variable(tf.zeros([9]))
    # we want the output to be calculated using softmax function.
    #softmax will compute the probability for each of avaliable output
    y = tf.nn.softmax(tf.matmul(hidden,W2)+b2)
    # the error function that we will try to minimize
    loss = tf.reduce_mean(tf.square(y - trainig_labels))
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #  y, trainig_labels, name='xentropy')
    #loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  
    #optimizer = tf.train.GradientDescentOptimizer(0.8)
    optimizer=tf.train.MomentumOptimizer(0.1,0.9)
   
    train = optimizer.minimize(loss)
    
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver()
    print "learning ......."
    import os
    if os.path.exists("model.ckpt"):
        print "restore session ......."
        saver.restore(sess, "model.ckpt")
    else:  
        loss_value=100
        step=0
        noise=0.0
        while loss_value>0 and step<5000:
            '''
            W=[]
            W.append(W1)
            W.append(W2)
            assign_ops,noise=add_noise(W,step,noise,sess)
            
            for assign_op in assign_ops:
                sess.run(assign_op)
            '''    
            sess.run(train,feed_dict={X:training_data,y_:trainig_labels})
            step=step+1
                    
            if step % 100 == 0:
                loss_value=sess.run(loss,feed_dict={X:training_data,y_:trainig_labels})
                print " in setp="+str(step)+" the loss value is "+str(loss_value)
            
        print "loss = 0 in step="+str(step)
        print "saving model ......."
        saver.save(sess,'model.ckpt')
    
    print "playing" 
    wins=0
    draws=0
    XWinningCount=0
    OWinningCount=0
    drawCaount=0
    for i in range(1000):
        print "start game number="+str(i)
        # initialize 3x3 tic tac toe board
        gameState = np.array([-1]*9).reshape(3,3)
            
        # initialize player number, move counter
        player = 1
        mvcntr = 1
    
        # initialize flag that indicates win
        noWinnerYet = True
        
    
        while move_still_possible(gameState) and noWinnerYet:
            # get player symbol
            name = symbols[player]
            print '%s moves' % name
    
            # if the player is x , then move in probabilistic move 
            if player==1:
                input_board=np.reshape(gameState,(9))
                input_board=np.reshape(encode_input(input_board, player),(1,18))
                #pos=np.argmax(sess.run(y,feed_dict={X:input_board}))
                indicies=np.argsort(sess.run(y,feed_dict={X:input_board})[0])[::-1]
                #pos=get_best_move(gameState,player)
                # go in reverse order
                for pos in indicies:
                    if gameState[pos/3][pos%3]==-1: # make sure it is empty
                        gameState[pos/3][pos%3]=player
                        break
            else:    
                gameState = move_at_random(gameState, player)
    
            # print current game state
            print_game_state(gameState)
            
            # evaluate game state
            if move_was_winning_move(gameState, player):
                print 'player %s wins after %d moves' % (name, mvcntr)
                wins=wins+1
                noWinnerYet = False
                if player==1:
                    XWinningCount=XWinningCount+1
                else:
                    OWinningCount=OWinningCount+1       
    
            # switch player and increase move counter
            player =(player+1)%2
            mvcntr +=  1
    
    
    
        if noWinnerYet:
            draws=draws+1
            print 'game ended in a draw' 
            
    print "x wins="+ str(XWinningCount)
    print "O wins ="+str(OWinningCount)  
    print "draws ="+str(draws)  
       
if __name__ == '__main__':
    main()
    '''
if __name__ == '__main__':
    wins=0
    draws=0
    XWinningCount=0
    OWinningCount=0
    drawCaount=0
    for i in range(1000):
        print "start game number="+str(i)
        # initialize 3x3 tic tac toe board
        gameState = np.zeros((3,3), dtype=int)
        for j in range(3):
            for k in range(3):
                gameState[j][k]=-1
            
        # initialize player number, move counter
        player = 1
        mvcntr = 1
    
        # initialize flag that indicates win
        noWinnerYet = True
        
    
        while move_still_possible(gameState) and noWinnerYet:
            # get player symbol
            name = symbols[player]
            print '%s moves' % name
    
            # if the player is x , then move in probabilistic move 
            if player==1:
                pos=get_best_move(gameState,player)
                if pos ==-1:
                    print "hh"
                print pos     
                gameState[pos/3][pos%3]=player
            else:    
                gameState = move_at_random(gameState, player)
    
            # print current game state
            print_game_state(gameState)
            
            # evaluate game state
            if move_was_winning_move(gameState, player):
                print 'player %s wins after %d moves' % (name, mvcntr)
                wins=wins+1
                noWinnerYet = False
                if player==1:
                    XWinningCount=XWinningCount+1
                else:
                    OWinningCount=OWinningCount+1       
    
            # switch player and increase move counter
            player =(player+1)%2
            mvcntr +=  1
    
    
    
        if noWinnerYet:
            draws=draws+1
            print 'game ended in a draw' 
            
    print "x wins="+ str(XWinningCount)
    print "O wins ="+str(OWinningCount)  
    print "draws ="+str(draws)  
    '''
           