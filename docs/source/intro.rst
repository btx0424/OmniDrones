Environment
===========

Challenging environments may have input and output with complex structures, 
such as multi-modal observations, where the commonly used OpenAI Gym interface becomes limited. 
To better work with the vectorized environments in Isaac Sim, which directly take in and output 
PyTorch Tensors in batches, we use `TorchRL <https://pytorch.org/rl/index.html>`_ 
and `TensorDict <https://pytorch.org/rl/tensordict/>`_ to build an efficient and flexible interface for OmniDrones.

.. note::

    hello

Specification
-------------
An environment's input/output specification is given by its ``observation_spec``, 
``action_spec``, ``reward_spec`` and optionally ``done_spec``.

.. code:: python

    from tensordict import TensorDictBase
    from omni_drones.envs.isaac_env import IsaacEnv
    
    env_class = IsaacEnv.REGISTRY["Hover"]
    env = env_class(cfg, headless=true)
    
    print(env.input_spec)
    print(env.output_spec)

.. code:: none

    CompositeSpec(
        _state_spec: CompositeSpec(
        , device=cuda, shape=torch.Size([4096])),
        _action_spec: CompositeSpec(
            agents: CompositeSpec(
                action: BoundedTensorSpec(
                    shape=torch.Size([4096, 1, 6]),
                    space=ContinuousBox(
                        minimum=Tensor(shape=torch.Size([4096, 1, 6]), device=cuda:0, dtype=torch.float32, contiguous=True), 
                        maximum=Tensor(shape=torch.Size([4096, 1, 6]), device=cuda:0, dtype=torch.float32, contiguous=True)),
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous), device=cuda, shape=torch.Size([4096])), device=cuda, shape=torch.Size([4096])), device=cuda, shape=torch.Size([4096]))
    CompositeSpec(
        _observation_spec: CompositeSpec(
            agents: CompositeSpec(
                observation: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1, 32]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous), device=cuda, shape=torch.Size([4096])),
            stats: CompositeSpec(
                return: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous),
                episode_len: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous),
                pos_error: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous),
                heading_alignment: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous),
                uprightness: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous),
                action_smoothness: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous), device=cuda, shape=torch.Size([4096])),
            info: CompositeSpec(
                drone_state: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1, 13]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous), device=cuda, shape=torch.Size([4096])), device=cuda, shape=torch.Size([4096])),
        _reward_spec: CompositeSpec(
            agents: CompositeSpec(
                reward: UnboundedContinuousTensorSpec(
                    shape=torch.Size([4096, 1, 1]),
                    space=None,
                    device=cuda,
                    dtype=torch.float32,
                    domain=continuous), device=cuda, shape=torch.Size([4096])), device=cuda, shape=torch.Size([4096])),
        _done_spec: CompositeSpec(
            done: DiscreteTensorSpec(
                shape=torch.Size([4096, 1]),
                space=DiscreteBox(n=2),
                device=cuda,
                dtype=torch.bool,
                domain=discrete), device=cuda, shape=torch.Size([4096])), device=cuda, shape=torch.Size([4096]))
        
.. code:: python

    print(env.reset())

output:

.. code:: none

    TensorDict(
        fields={
            agents: TensorDict(
                fields={
                    observation: Tensor(shape=torch.Size([4096, 1, 32]), device=cuda:0, dtype=torch.float32, is_shared=True)},
                batch_size=torch.Size([4096]),
                device=cuda,
                is_shared=True),
            done: Tensor(shape=torch.Size([4096, 1]), device=cuda:0, dtype=torch.bool, is_shared=True),
            info: TensorDict(
                fields={
                    drone_state: Tensor(shape=torch.Size([4096, 1, 13]), device=cuda:0, dtype=torch.float32, is_shared=True)},
                batch_size=torch.Size([4096]),
                device=cuda,
                is_shared=True),
            progress: Tensor(shape=torch.Size([4096]), device=cuda:0, dtype=torch.float32, is_shared=True),
            stats: TensorDict(
                fields={
                    action_smoothness: Tensor(shape=torch.Size([4096, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
                    episode_len: Tensor(shape=torch.Size([4096, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
                    heading_alignment: Tensor(shape=torch.Size([4096, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
                    pos_error: Tensor(shape=torch.Size([4096, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
                    return: Tensor(shape=torch.Size([4096, 1]), device=cuda:0, dtype=torch.float32, is_shared=True),
                    uprightness: Tensor(shape=torch.Size([4096, 1]), device=cuda:0, dtype=torch.float32, is_shared=True)},
                batch_size=torch.Size([4096]),
                device=cuda,
                is_shared=True)},
        batch_size=torch.Size([4096]),
        device=cuda,
        is_shared=True)


Interaction and Stepping Logic
------------------------------

``IsaacEnv.step`` accepts a ``tensordict`` at each time step which contains 
the input the environment (actions, typically). 

.. code:: python

    def policy(tensordict: TensorDictBase):
        # a dummy policy
        tensordict.update(env.action_spec.zero())
        return tensordict
    
    tensordict = env.reset()
    tensordict = policy(tensordict)
    tensordict = env.step(tensordict)
    
    print(tensordict) # the first transition

where ``env.step`` executes roughly the following logic:

.. code:: python

    # omni_drones/envs/isaac_env.py

    class IsaacEnv():
        ...
        def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
            self._pre_sim_step(tensordict) # apply actions, custom physics
            for substep in range(self.substeps):
                self.sim.step(self._should_render(substep))
            self._post_sim_step(tensordict) # state clipping, post processing, etc.
            self.progress_buf += 1
            tensordict = TensorDict({"next": {}}, self.batch_size)
            tensordict["next"].update(self._compute_state_and_obs())
            tensordict["next"].update(self._compute_reward_and_done())
            return tensordict

output:

.. code:: none

    ...

Data Collection
---------------

.. code:: python

    from omni_drones.utils.torchrl import SyncDataCollector

    frames_per_batch = env.num_envs * 128
    
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        device=env.device,
        return_same_td=True,
    )

    for i, data in enumerate(collector):
        # training and logging logic here
        break
    
    print(data)
    

Creating New Tasks
------------------