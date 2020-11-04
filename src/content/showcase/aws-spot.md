+++
date = "2019-02-14"
title = "Affordable Deep Learning with automated AWS Spot Instances"
showonlyimage = false
draft = false
image = "https://miro.medium.com/max/573/0*Ws2g7yqu0dYEcmwO.jpg"
weight = 4
+++

The easiest way to get the cheapest Amazon instances for your deep learning projects.
<!--more-->

![img](https://miro.medium.com/max/573/0*Ws2g7yqu0dYEcmwO.jpg)


- Originally published on [Medium](https://medium.com/@sdhnshu/pro-deep-learning-setup-at-90-off-e9e68f5e84ec)
- All code available on [Github](https://github.com/sdhnshu/Deep-Jupyter-Portal)

### Introduction

__Amazon Spot instances__ offer __spare compute capacity__ available at __steep discounts__. You can check out the pricing [here](https://aws.amazon.com/ec2/spot/pricing/). The only catch is, these instances can be taken away from you anytime.

For a person who wants to focus on deep learning and not on making spot instances reliable, I’ve automated the process and set up the project here: *[github.com/Deep-Jupyter-Portal](https://github.com/sdhnshu/Deep-Jupyter-Portal)*

Spot instances can be launched with any AMI (Amazon Machine Image). We’ll be choosing an AMI with all deep learning libraries installed. The package includes some storage (in our case 60–75GB) and I am choosing a g2.2xlarge machine which has 8 core CPU, 15 GB RAM, and NVIDIA GK104 GPU with 8GB graphics memory.

In the time of my using spot instances, I’ve seen them stable for days. But we’ll setup another persistent storage (20 GB) that will never be taken away from us, later in this post. All our data will be saved in this volume.

#### What would you get?

Once your preferences are setup, with __one command__, you can spawn a cheap AWS instance. With another you can launch a __jupyter notebook__ on it, which automatically opens up in your browser. It also automatically syncs back the data from the server to your machine.

#### 4 step installation

We’ll be using a library called portal-gun (v 0.2.0). It can automate the reservation of a spot instance. Check out its [documentation](https://portal-gun.readthedocs.io/en/stable/) for more details.

Clone this [github.com/Deep-Jupyter-Portal](https://github.com/sdhnshu/Deep-Jupyter-Portal) and setup a python 2.7 environment. Install the requirements.txt in the environment. To setup the config for portal-gun, we need to setup four things in AWS.

#### Step 1. IAM user

Portal-gun will need some access keys for communicating to AWS. Create a new IAM user [here](https://console.aws.amazon.com/iam/home), and give it programmatic access. You’ll see the following screen.

![Create Iam user](https://miro.medium.com/max/573/1*brtvmLnSuqz1GQoKu4ancg.png)

To attach a policy, create a new one with the following permissions and attach it to the user. These are all the permissions portal-gun needs to access the servers.

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iam:PassRole",
                "ec2:DescribeAccountAttributes",
                "ec2:DescribeAvailabilityZones",
                "ec2:DescribeSubnets",
                "ec2:CreateVolume",
                "ec2:ModifyVolume",
                "ec2:AttachVolume",
                "ec2:DetachVolume",
                "ec2:DeleteVolume",
                "ec2:DescribeVolumes",
                "ec2:DescribeVolumeStatus",
                "ec2:DescribeVolumeAttribute",
                "ec2:DescribeVolumesModifications",
                "ec2:RequestSpotFleet",
                "ec2:CancelSpotFleetRequests",
                "ec2:RequestSpotInstances",
                "ec2:CancelSpotInstanceRequests",
                "ec2:ModifySpotFleetRequest",
                "ec2:ModifyInstanceAttribute",
                "ec2:DescribeSpotFleetRequests",
                "ec2:DescribeSpotInstanceRequests",
                "ec2:DescribeSpotFleetInstances",
                "ec2:DescribeSpotPriceHistory",
                "ec2:DescribeSpotFleetRequestHistory",
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus",
                "ec2:DescribeInstanceAttribute",
                "ec2:CreateTags",
                "ec2:DeleteTags",
                "ec2:DescribeTags"
            ],
            "Resource": "*"
        }
    ]
}
```

You’ll get an access key and a secret key after the user is created. Make a file _~/.portal-gun/config.json_ if not present and add/replace the following lines to it.

```json
{
    "aws_region": "us-east-1",
    "aws_access_key": "AKIAXXXXXXXXX7F6QQ",
    "aws_secret_key": "Ji4esAyXXXXXXXXXXXXXXXXXXXXXXXXXtrQBjcg"
}
```

I’ll be working in the US East (N Virginia) region, because they have all the GPU machines one can think of.

#### Step 2. RSA keys

To ssh into the server, we need to add your laptop’s RSA public key (~/.ssh/id_rsa.pub) to the key-pairs in the EC2 tab. Note down the key pair name.

![Rsa keys](https://miro.medium.com/max/573/1*gwxJfptEoI-APfwDPDpV_g.png)

#### Step 3. Security Group

Create a new security group using the network and security section on the left. Make a group with 22 (ssh) and 8888 (jupyter) ports open. This would allow communication through these ports. Note down the security group id.

![sec group](https://miro.medium.com/max/573/1*X9QsJLMxbzC3jmsW7WLYvA.png)

#### Step 4. Persistent volume

The volumes with the AMI would be destroyed when the spot instance is taken away from you. Create a volume in the us-east-1a region using the Elastic Block Store section on the left. This would be used for our permanent storage. Note down the volume id.

![volume](https://miro.medium.com/max/573/1*wgOYPAioOK35WLVfR6DOYw.png)

#### Step 5. AMI (optional)

I’ve chosen the deep learning AMI v21.0. It has all deep learning frameworks installed with CUDA and CuDNN. Note down the AMI id.

![ami](https://miro.medium.com/max/573/1*HEGhWvHprMN7pT9imtw30g.png)

__*If you are using the same AMI as I am, you can jump to the next section.*__

To find out what mount points are free in the AMI, we need to first start an instance just with just the AMI and run `lsblk`.

![lbsk](https://miro.medium.com/max/316/1*PejpOV_FJ9QooVSSaMrjUw.png)

Here we can see 2 disks xvda and xvdb are mounted. We can use generally use xvd(g-j) for our usage. If it says sda, sdb in your AMI, choose something within sd(g-j). I’ve chosen xvdg.

#### Verifying the config file

Open up the deep_gpu.json file from the repo. Check all marked values and change them to your needs. Put your account number in the morphed location in iam_fleet_role. You can find your account number here on the top left.

The channels block is for a real-time sync of data between your local machine and the server. Direction in specifies server to local and out stands for local to server.

Remember to keep both *local_paths* different.

![deepgpu.json](https://miro.medium.com/max/573/1*DusmttvQI49TXsoZdOK71A.png)

### Opening the portal

To start the portal, we use the following command from the path where deep_gpu.json is located.

```cmd
portal open deep_gpu
```

![open the portal](https://miro.medium.com/max/484/1*Po9bZb0X8gg3Qdoz39MnXw.png)

For me it takes around 4 minutes to find and setup the machine. To interact with the AWS instance, use the following commands.

```cmd
portal open deep_gpu
portal info deep_gpu
portal ssh deep_gpu
portal close deep_gpu
```

### Starting the jupyter notebook

```cmd
python start_jupyter.py
```

This will run the jupyter notebook on the machine and open it in your browser. The jupyter service might take a while to start. Refresh the browser in a while.

Starting the jupyter notebook also starts the sync process and syncs both folders as anything changes in them.

After you are done working, just use the following command to terminate the spot instance.

```cmd
portal close deep_gpu
```

Reach out to me in the comments if you have any issues.


#### References
- [Vadim Fedorov](https://hackernoon.com/@coderik) for making the library (portal-gun). You can check out his blog post [here](https://hackernoon.com/deep-learning-on-amazon-ec2-spot-instances-without-the-agonizing-pain-4cedf9b129c4).