### ERROR1:   
+ Condition: Detectron2, pytorch, docker, balloon dataset    
+ Error Description:      
RuntimeError: DataLoader worker (pid 8554) is killed by signal: 
Bus error. It is possible that dataloader's workers are out of shared memory. 
Please try to raise your shared memory limit.
+ Analysis: 这是因为在起docker容器的时候，设置的/dev/shm只有64M。多线程加载数据时需要把一部分数据放到这部分内存跑就会报错了
+ Solution1: 在训练模型时，将配置参数设置为：cfg.DATALOADER.NUM_WORKERS = 1
+ Solution2：在docker run的时候设置参数`--ipc=host`或者`--shm-size="4g"`，[参考](https://www.zhihu.com/question/40125229)。
+ Solution3：关闭容器-->找到容器-->cd /var/lib/Docker/containers/-->修改hostconfig.json-->修改shmsize-->修改shmsize，[参考1](https://blog.csdn.net/u013985291/article/details/87778410),[参考2](https://www.jianshu.com/p/4398bdb9e2d2),[参考3](https://blog.csdn.net/shmily_lsl/article/details/81166951)

