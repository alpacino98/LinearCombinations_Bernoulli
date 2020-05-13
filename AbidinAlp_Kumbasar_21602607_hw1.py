import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import sys

question_no = sys.argv[1]

if int(question_no) == 1:
    # ## Question 1

    # In[2]:


    a = np.array([[1,0,-1,2],[2,1,-1,5],[3,3,0,9]])
    print("Matrix A is: \n")
    print(a)
    print()
    b = np.array([1,4,9])
    print("Matrix B is: \n")
    print(b)


    # ### Question 1  A

    # In[3]:


    alfa = np.random.rand()
    beta = np.random.rand()
    x_a = np.array([alfa - 2 * beta, - alfa - beta, alfa, beta])
    result_1_a = a.dot(x_a)
    print(result_1_a)


    # ### Question 1 B

    # In[4]:


    alfa = 0
    beta = 1
    x_p_b = np.array([1 + alfa - 2 * beta, 2 - alfa - beta, alfa, beta])
    result_1_b = a.dot(x_p_b)
    print("Xn particular solution matrix is: " + str(x_p_b))
    print(result_1_b)


    # ### Question 1 C

    # In[5]:


    alfa = np.random.rand()
    beta = np.random.rand()
    x_p_c = np.array([1 + alfa - 2 * beta, 2 - alfa - beta, alfa, beta])
    result_1_c = a.dot(x_p_c)
    print(result_1_c)


    # ### Question 1 D

    # In[6]:


    u, s, v_t = np.linalg.svd(a)
    s_h = np.zeros(a.shape)
    x,y = s_h.shape

    for i in range(x):
        for j in range(y):
            if i == j and i != 2:
                s[i] = 1/s[i]
                s_h[i,j] = s[i]
    s_p = s_h

    inv_A = v_t.T.dot(s_p.T)
    pinv_A = inv_A.dot(u.T)

    print("Psuedo Inverse of A is : \n", pinv_A)

    a_app = a.dot(pinv_A).dot(a)

    print("Approximation of A using Psuedo Inverse: \n", a_app)


    # In[7]:


    pinv_A_2 = np.linalg.pinv(a)
    print(pinv_A_2)


    # ### Question 1 E

    # In[8]:


    free_vars = [(0,0),(0,2),(0,1/2),(-1,0),(2,0),(1,1)]
    count = 1
    result_holder = []
    a = np.array([[1,0,-1,2],[2,1,-1,5],[3,3,0,9]])

    for alpha, b in free_vars:
        x_sparese = np.array([1 + alpha - 2 * b, 2 - alpha - b, alpha, b])
        result_holder.append(a.dot(x_sparese))
        print("Sparsest x vector found in calculation " + str(count) + " is: ")
        print(x_sparese)
        print("Sparsest solution " + str(count) + ": ")
        print(result_holder[count - 1])
        
        count += 1


    # ### Question 1 F

    # In[9]:


    alpha = 10/17
    beta = 13/17

    a_f = np.array([[1,0,-1,2],[2,1,-1,5],[3,3,0,9]])
    x_f = np.array([1 + alpha - 2 * beta, 2 - alpha - beta, alpha, beta])

    print("Least Norm solution vector of x is :" )
    print(x_f)
    print("Result constructed by least norm vector of x is:")
    print(a_f.dot(x_f))

elif int(question_no) == 2:
    # ### Question 2


    # ### Question 2 A

    # In[11]:


    def bern_like(p, tot_num, active):
        like_holder = (p ** active) * ((1-p) ** (tot_num - active)) * comb(tot_num, active)
        return like_holder


    # In[12]:


    x = np.arange(0, 1, 0.001)
    act_lan = 103
    lan = 869
    act_nolan = 199
    no_lan = 2353

    like_act_l = bern_like(x, lan, act_lan)
    like_act_nolan = bern_like(x, no_lan, act_nolan)


    # In[13]:


    plt.figure()
    plt.xlabel("Probability of activation given language")
    plt.ylabel("Likelihood found in given probability")
    plt.title("Question 2 A 1st Graph Likelihood vs. Activation probability given language")
    plt.xlim(0,250) 
    plt.xticks(np.arange(0,251,50), (0, 0.05, 0.1, 0.15, 0.2, 2.5))
    plt.bar(np.arange(len(x)),like_act_l)


    # In[14]:


    plt.figure()
    plt.xlabel("Probability of activation not given language")
    plt.ylabel("Likelihood found in not given probability")
    plt.title("Question 2 A 1st Graph Likelihood vs. Activation probability not given language")
    plt.xlim(0,150) 
    plt.xticks(np.arange(0,151,50), (0, 0.05, 0.1, 0.15))
    plt.bar(np.arange(len(x)),like_act_nolan)


    # ### Question 2 B

    # In[15]:


    max_xl = x[np.argmax(like_act_l)]
    print("Maximized likelihood function value of activation given language is :")
    print(max_xl)

    max_xnl = x[np.argmax(like_act_nolan)]
    print("Maximized likelihood function value of activation given language is :")
    print(max_xnl)


    # ### Question 2 C

    # In[16]:


    pxld = 1 / len(x) * like_act_l / np.sum(1/len(x) * like_act_l)

    pxnld = 1/ len(x) * like_act_nolan / np.sum(1 / len(x) * like_act_nolan) 


    # In[17]:


    plt.figure()
    plt.xlabel("Probability Active Given Language")
    plt.ylabel("Discrete Posterior Distribution of P(X|Data)")
    plt.xlim(0,200)
    plt.xticks(np.arange(0,201,50), (0, 0.05, 0.1, 0.15, 0.2))
    plt.bar(np.arange(len(x)),pxld)


    # In[18]:


    plt.figure()
    plt.xlabel("Probability Active Given Not Language")
    plt.ylabel("Discrete Posterior Distribution of P(X|Data)")
    plt.xlim(0,200)
    plt.xticks(np.arange(0,201,50), (0, 0.05, 0.1, 0.15, 0.2))
    plt.bar(np.arange(len(x)),pxnld)


    # In[19]:


    pxld_cum = []
    pxnld_cum = []
    summer_l = 0
    summer_n = 0

    for i in range(len(x)):
        summer_l = summer_l + pxld[i]
        pxld_cum.append(summer_l)


        summer_n = summer_n + pxnld[i]
        pxnld_cum.append(summer_n)
        
    pxld_cum = np.array(pxld_cum)
    pxnld_cum = np.array(pxnld_cum)


    # In[20]:


    plt.figure()
    plt.xticks(np.arange(0, 1001, 200), (0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel("Probability Active Given Language")
    plt.ylabel("Cummulative Posterior Distribution of P(X|Data)")
    plt.bar(np.arange(len(x)), pxld_cum)

    plt.figure()
    plt.xlim(0,250)
    plt.xticks(np.arange(0, 251, 50), (0, 0.05, 0.1, 0.15, 0.2, 0.25))
    plt.xlabel("Probability Active Given Language")
    plt.ylabel("Cummulative Posterior Distribution of P(X|Data)")
    plt.bar(np.arange(len(x)), pxld_cum)


    # In[21]:


    plt.figure()
    plt.xticks(np.arange(0, 1001, 200), (0, 0.2, 0.4, 0.6, 0.8, 1))
    plt.xlabel("Probability Active Given Not Language")
    plt.ylabel("Cummulative Posterior Distribution of P(X|Data)")
    plt.bar(np.arange(len(x)), pxnld_cum)

    plt.figure()
    plt.xlim(0,250)
    plt.xticks(np.arange(0, 251, 50), (0, 0.05, 0.1, 0.15, 0.2, 0.25))
    plt.xlabel("Probability Active Given Not Language")
    plt.ylabel("Cummulative Posterior Distribution of P(X|Data)")
    plt.bar(np.arange(len(x)), pxnld_cum)


    # In[22]:


    up_bound_pos_xl = np.argmin(np.abs(pxld_cum - 0.975))
    low_bound_pos_xl = np.argmin(np.abs(pxld_cum - 0.025))
    print("Upper Confidence Bound of 95% of xl is: " + str(x[up_bound_pos_xl]))
    print("Lower Confidence Bound of 95% of xl is: " + str(x[low_bound_pos_xl]))
        
    up_bound_pos_xnl = np.argmin(np.abs(pxnld_cum - 0.975))
    low_bound_pos_xnl = np.argmin(np.abs(pxnld_cum - 0.025))
    print("Upper Confidence Bound of 95% of xnl is: " + str(x[up_bound_pos_xnl]))
    print("Lower Confidence Bound of 95% of xnl is: " + str(x[low_bound_pos_xnl]))
        


    # ### Question 2 D



    # In[25]:


    joint_prob = np.zeros((1000,1000))


    # In[26]:


    for i in range(pxnld.__len__()):
        for j in range(pxld.__len__()):
            joint_prob[i,j] = pxld[j] * pxnld[i]
            
    print("Max Joint prob is in index:" + str(np.argmax(joint_prob)))


    # In[27]:


    plt.figure()
    plt.title("Joint Posterior Distribution of XL and XNL")
    plt.xticks(np.arange(len(x), step=100), 
            (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    plt.yticks(np.arange(len(x), step=100), 
            (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    plt.xlabel("xnl")
    plt.ylabel("xl")
    plt.imshow(joint_prob)
    plt.show()


    # In[32]:


    summer_bigger = 0
    summer_lower = 0

    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if pxld[i] > pxnld[j]:
                summer_bigger = summer_bigger + joint_prob[i,j]
            else:
                summer_lower = summer_lower + joint_prob[i,j]
    print("P(Xl > Xnl | Data)" + str(summer_bigger))
    print("P(Xl < Xnl | Data)" + str(summer_lower))


    # ### Question 2 E

    # In[33]:


    pro_lan = 0.5
    pro_no_lan = 0.5
    print(max_xl * pro_lan  *  max_xnl * pro_no_lan)
    print((max_xl * 0.5))

    p_lang_act = (max_xl * 0.5) / ( max_xl * pro_lan  + max_xnl * pro_no_lan)
    print("P(language | activation): " + str(p_lang_act))


else:
    print("No question is matched with the entered number, please enter again!!")




