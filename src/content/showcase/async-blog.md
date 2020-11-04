+++
date = "2018-07-30"
title = "Asynchronous Python App Architecture"
showonlyimage = false
draft = false
image = "https://miro.medium.com/max/700/0*y_3JGr5VRvkcAcjE.png"
weight = 5
+++

Learn how to choose the correct architecture for your application.
<!--more-->

![img](https://miro.medium.com/max/700/0*y_3JGr5VRvkcAcjE.png)

- Originally published on [Medium](https://medium.com/cowrks/asynchronous-python-app-architecture-5395d5338c4a)

### Introduction
Choosing an application architecture greatly depends on the scale you are building it for.

> 1x Monolith
>
> 10x Micro service
>
> 100x Asynchronous Micro service
>
> 1000x Distributed

In this post, we’ll be deep-diving into how to build an asynchronous python api with async db connections.

We’ll be using sanic (v0.7) as the api framework and asyncpg (v0.15) to connect to a postgres db. You can use any async db framework you prefer.

---
### How we use async?

We started using sanic after we had experimented with various other api frameworks like falcon, flask, tornado and twisted. In a few months, python added asyncio to its framework. Libraries started popping up using this async feature. We found this framework when it had around 500 stars on github. And the community grew so fast and at the time of this writing it has shot up to over 10k stars. We started using it on our production servers and have yet to encounter any serious issues.

Due to that switch, we had to use async db frameworks, async file readers, async requests package and many more async packages. We have started building async libraries for our apps and are quite satisfied by the performance.

---
### Is python purely asynchronous?

No. The most that we can get with python is context switching between different tasks so that the thread is being used to its optimum processing capacity. (Don’t worry if you didn’t get this, we’ll come back to this later.)

---
### How does async in python work?

The concept of asynchronous programming was introduced into python 3.4. It was introduced very late due to a fundamental python concept — Global Interpreter Lock (GIL) which blocked it from being asynchronous. If you do want to elevate the api scale to a massive level, you might have to resort to some other approach for instance having a distributed cluster, building an api that supports pure async eg: c, cpp.

Almost all asynchronous python frameworks use python’s standard library: asyncio underneath. All of these async libraries have similar architectures.

#### Event loop

The central part of async is an event loop. As the name suggests, it is a loop. One python event loop runs on one OS thread. So only one calculation will be executed by the processor at a time. You can add tasks to this loop and it executes them on a FIFO fashion. So it’s really important that your tasks have a trigger (more about it later) that passes the control back to the loop when it is doing an input/output function that does not require processing power. This transfer of control is the central theme of any async framework. Once the control is back at the loop, it can execute the other tasks in the loop.

Here’s how you can create an event loop using asyncio:

```python
loop = asyncio.get_event_loop()
loop.run_until_complete(tasks())
```

The command in the second line is directing the loop to run until all the tasks are complete. The above function tasks is an async function. We’ll explain how to define an async function later on.

The loop controls what task gets executed when, and does the context switching between different tasks.

#### Defining async functions

Here’s the syntax of async function:
```python
async def function_name():
    resp = await some_other_async_function()
    return resp
```
This function is added to an event loop to be executed as a task.

Here you can see two new things async and await. Async means the function is asynchronous. When the interpreter reaches the Await part, it passes the control back to the event loop (the trigger) and waits for some_other_async_function to complete. some_other_async_function is a function that doesn’t require computation, it can be a input/output operation like a db fetch or a db input. So while that db query is being executed, other tasks in the event loop can be executed. And when the response comes back from the db function, it goes back to the resp variable and is returned.

So there’s nothing special about this function except that it can pause itself and give others the time to execute. It doesn’t block the thread when not needed.

So an async function can call another async function; it can also call a sync function. But remember that it will block the thread till it completes so it must be using some computation. Also every async function will be awaited by the one who called it.

---

### Sanic

Sanic is a api framework that uses [uvloop](https://magic.io/blog/uvloop-blazing-fast-python-networking/) as its event loop. Uvloop is what makes sanic blazingly fast. Its around __2x faster than any nodejs__ async server. Almost __5x faster than tornado or twisted__ (python api frameworks) and has half the latency of any other servers (and is obviously around 10x faster than any sync api framework like falcon, flask, django). Uvloop is written in Cython and is built on top of a asynchronous library written in C.

Let’s check out an example:
```python
app = sanic.Sanic()app.add_route(root_func, '/', methods=['GET'])
app.add_route(users_func, '/users/<uid>', methods=['GET'])if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=2)
```
Here, `root_func`, `users_func` are async functions.
```python
async def root_func(request):
    resp = await do_whatever()
    return sanic.response.json({'resp': resp})async def users_func(request, uid):
    resp = await do_whatever()
    return sanic.response.json({'resp': resp})
```
In a sync api framework, the app can take only one request at a time, process it completely and then move on to other requests. And if that process is an i/o operation and takes a lot of time, the app will be stuck till it is completed. But here, it can be awaited and the app can work on other requests while the i/o operation returns a response.

Hence we are not executing things in parallel, just switching between tasks when free.

To add some more speed to the app, we can increase the number of workers while creating the app. But remember, a higher number of workers doesn’t necessarily mean faster speed (processors are also limitedly async.)

---
### Asyncpg

To use asyncpg or any other async library with sanic, we need to perform one more step. Sanic does not use the default asyncio loop underneath. It uses another loop called uvloop which is a faster version of the asyncio loop. So when we define the asyncpg connection, we need to connect that task to the uvloop which is already running instead of creating another event loop and adding tasks to that. In order to do that you need access to the uvloop object. Here’s how you do it:
```python
app = sanic.Sanic()

@app.listener('before_server_start')
async def before_start(app, uvloop):
    db = await asyncpg.connect(postgresdb, loop=uvloop)

app.add_route(root_func, '/', methods=['GET'])
app.add_route(users_func, '/users/<uid>', methods=['GET'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=2)
```
Now this db variable can be used with your project and is adding tasks in the uvloop. The function before_start is a sanic special function that gives you access to things before the server has started.

---
### Bonus points
#### Semaphore

To avoid the async api from overloading due to a Denial Of Service (DOS) attack, you can add a semaphore to limit the number of simultaneous api requests that are computed by doing the following:
```python
app = sanic.Sanic()

@app.listener('before_server_start')
async def before_start(app, uvloop):
    sem = await asyncio.Semaphore(100, loop=uvloop)

app.add_route(root_func, '/', methods=['GET'])
app.add_route(users_func, '/users/<uid>', methods=['GET'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=2)

-------------------------------------------

async def users_func(request, uid):
    async with sem:
        resp = await do_whatever()
        return sanic.response.json({'resp': resp})

async def root_func(request):
    async with sem:
        resp = await do_whatever()
        return sanic.response.json({'resp': resp})
```
Here I’ve set a semaphore for 100 and attached it to the uvloop. And every api function will be called only when the semaphore lock is open.

#### Async classes

Classes with async functions are defined as:
```python
class Class_name:
    def __init__(self):
        pass
    async def func1(self):
        pass
    async def func2(self):
        pass
```

They are initialized normally but the functions will be awaited because they are async.

```python
c = Class_name()
resp = await c.func1()
resp = await c.func2()
```

-------

### Conclusion

Asynchronous application architecture is fairly easy to understand and is used widely in all languages especially large scaling apps. It is also a useful tool to have in your arsenal if you are looking towards upgrading your coding game.