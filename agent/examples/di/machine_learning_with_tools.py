import asyncio

from metagpt.actions import WriteAnalysisCode
from metagpt.roles.di.data_interpreter import DataInterpreter
from metagpt.utils.recovery_util import save_history
from shared_queue import log_execution


# async def main(requirement: str):
#     role = DataInterpreter(use_reflection=True, tools=["<all>"])
#     # role = DataInterpreter(use_reflection=True)
#     await role.run(requirement)
#     save_history(role=role)



async def main_generator_with_interpreter(interpreter: DataInterpreter, requirement: str, user_id: str):
    await log_execution("#### 🔥Starting main function\n", user_id)
    role = interpreter  # 假设 'interpreter' 类似于 'role'
    role.set_actions([WriteAnalysisCode])
    role._set_state(0)
    await role.run(requirement, user_id)
    save_history(role=role)
    await log_execution("#### Finished main function😊\n", user_id)

async def main_generator(requirement1: str):
    await log_execution("#### 🔥Starting main function\n", "1")
    # 创建两个DataInterpreter实例
    role1 = DataInterpreter(use_reflection=True, tools=["<all>"])
    # role2 = DataInterpreter(use_reflection=True, tools=["<all>"])
    # 同时运行两个实例
    await asyncio.gather(
        role1.run(requirement1, user_id="1"),
        # role2.run(requirement2)
    )
    
    # 设置两个实例的actions
    # role1.set_actions([WriteAnalysisCode])
    # role2.set_actions([WriteAnalysisCode])
    
    # # 重置状态
    # role1._set_state(0)
    # role2._set_state(0)
    
    # 保存历史记录
    save_history(role=role1)
    # save_history(role=role2)
    
    await log_execution("#### Finished main function😊\n", "1")


if __name__ == "__main__":
    requirement1 = "The tasks you want the econometric agent to complete"
    asyncio.run(main_generator(requirement1))
