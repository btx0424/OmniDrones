
- /home/hesicheng/OmniDrones/omni_drones/learning/mappo.py `Actor.forward` modified

---

# PingPong2v2

## Introduction

- The first two drones as group_0 and the last two as group_1
- group_0 is the agent and group_1 is the fixed opponent


## Information

- SyncDataCollector.__init__ costs about 0.1s

## To Do
先测试一下SyncDataCollector构造的开销。目测不是很大。那样就可以写个policy wrapper套住两个team的policy，一个fixed一个要训。每次更换对手重新构造一个即可。
