#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import pickle
import re
import textwrap
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .tool_validation import validate_tool_attributes
from .tools import Tool
from .utils import BASE_BUILTIN_MODULES, instance_to_source


try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class AgentCallInterruption(Exception):
    def __init__(
        self,
        agent_name: str,
        args: tuple,
        kwargs: dict,
        line_number: Optional[int] = None,
    ):
        self.agent_name = agent_name
        self.args = args
        self.kwargs = kwargs
        self.line_number = line_number
        super().__init__(
            f"Sub-agent call interruption for '{agent_name}' at line {line_number}"
        )


class E2BExecutor:
    def __init__(
        self,
        additional_imports: List[str],
        tools: List[Tool],
        logger,
        managed_agents: Optional[Dict[str, Any]],
    ):
        self.logger = logger
        try:
            from e2b_code_interpreter import Sandbox
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'e2b' extra to use E2BExecutor: `pip install "smolagents[e2b]"`"""
            )
        self.logger = logger
        self.logger.log("Initializing E2B executor, hold on...")

        self.managed_agents = managed_agents or {}
        self.custom_tools = {}
        self.final_answer = False
        self.final_answer_pattern = re.compile(r"final_answer\((.*?)\)")
        self.sbx = Sandbox()  # "qywp2ctmu2q7jzprcf4j")
        # TODO: validate installing agents package or not
        # print("Installing agents package on remote executor...")
        # self.sbx.commands.run(
        #     "pip install git+https://github.com/huggingface/smolagents.git",
        #     timeout=300
        # )
        # print("Installation of agents package finished.")
        additional_imports = list(set(additional_imports + ["smolagents", "sys"]))
        if len(additional_imports) > 0:
            execution = self.sbx.commands.run(
                "pip install " + " ".join(additional_imports)
            )
            if execution.error:
                raise Exception(f"Error installing dependencies: {execution.error}")
            else:
                logger.log(f"Installation of {additional_imports} succeeded!", 0)

        tool_codes = []
        for tool in tools:
            validate_tool_attributes(tool.__class__, check_imports=False)
            tool_code = instance_to_source(tool, base_cls=Tool)
            tool_code = tool_code.replace("from smolagents.tools import Tool", "")
            tool_code += f"\n{tool.name} = {tool.__class__.__name__}()\n"
            tool_codes.append(tool_code)

        tool_definition_code = "\n".join(
            [f"import {module}" for module in BASE_BUILTIN_MODULES]
        )
        tool_definition_code += textwrap.dedent(
            """
        class Tool:
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

            def forward(self, *args, **kwargs):
                pass # to be implemented in child class
        """
        )
        tool_definition_code += "\n\n".join(tool_codes)
        for agent_name in self.managed_agents.keys():
            tool_definition_code += "\n\n" + textwrap.dedent(
                f"""
                def {agent_name}(*args, **kwargs):
                    line_no = sys._getframe(1).f_lineno
                    raise AgentCallInterruption("sub_agent", args, kwargs, line_number=line_no)
                """
            )

        tool_definition_execution = self.run_code_raise_errors(tool_definition_code)
        self.logger.log(tool_definition_execution.logs)

    def run_code_raise_errors(self, code: str):
        if self.final_answer_pattern.search(code) is not None:
            self.final_answer = True
        execution = self.sbx.run_code(code)
        if execution.error:
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            logs = execution_logs
            logs += "Executing code yielded an error:"
            logs += execution.error.name
            logs += execution.error.value
            logs += execution.error.traceback
            raise ValueError(logs)
        return execution

    def __call__(self, code_action: str, additional_args: dict) -> Tuple[Any, Any]:
        if len(additional_args) > 0:
            # Pickle additional_args to server
            import tempfile

            with tempfile.NamedTemporaryFile() as f:
                pickle.dump(additional_args, f)
                f.flush()
                with open(f.name, "rb") as file:
                    self.sbx.files.write("/home/state.pkl", file)
            remote_unloading_code = """import pickle
import os
print("File path", os.path.getsize('/home/state.pkl'))
with open('/home/state.pkl', 'rb') as f:
    pickle_dict = pickle.load(f)
locals().update({key: value for key, value in pickle_dict.items()})
"""
            execution = self.run_code_raise_errors(remote_unloading_code)
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            self.logger.log(execution_logs, 1)

        while True:
            try:
                execution = self.run_code_raise_errors(code_action)
                break
            except AgentCallInterruption as handoff:
                self.logger.log(
                    f"Running sub-agent '{handoff.agent_name}' locally with args {handoff.args} and kwargs {handoff.kwargs}",
                    level=0,
                )
                result = self.managed_agents[handoff.agent_name](
                    *handoff.args, **handoff.kwargs
                )
                additional_args[handoff.agent_name] = result

                # Re-upload the updated state to the sandbox.
                import tempfile

                with tempfile.NamedTemporaryFile() as f:
                    pickle.dump(additional_args, f)
                    f.flush()
                    with open(f.name, "rb") as file:
                        self.sbx.files.write("/home/state.pkl", file)
                self.logger.log(
                    f"Sub-agent '{handoff.agent_name}' executed locally; resuming sandbox execution.",
                    level=0,
                )
                # Replace the call to the sub-agent with the result of the local execution.
                lines = code_action.splitlines()
                if handoff.line_number is not None and 0 < handoff.line_number <= len(
                    lines
                ):
                    variable = lines[handoff.line_number - 1].split("=")[0].strip()
                    assignment_line = f"{variable} = {repr(result)}"
                    code_action = "\n".join(
                        [assignment_line] + lines[handoff.line_number - 1 :]
                    )
                else:
                    raise ValueError(
                        f"Invalid line number {handoff.line_number} for code action:\n{code_action}"
                    )
                # Execute the rest of the code action.
                continue

        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        if not execution.results:
            return None, execution_logs, self.final_answer
        else:
            for result in execution.results:
                if result.is_main_result:
                    for attribute_name in ["jpeg", "png"]:
                        if getattr(result, attribute_name) is not None:
                            image_output = getattr(result, attribute_name)
                            decoded_bytes = base64.b64decode(
                                image_output.encode("utf-8")
                            )
                            return (
                                Image.open(BytesIO(decoded_bytes)),
                                execution_logs,
                                self.final_answer,
                            )
                    for attribute_name in [
                        "chart",
                        "data",
                        "html",
                        "javascript",
                        "json",
                        "latex",
                        "markdown",
                        "pdf",
                        "svg",
                        "text",
                    ]:
                        if getattr(result, attribute_name) is not None:
                            return (
                                getattr(result, attribute_name),
                                execution_logs,
                                self.final_answer,
                            )
            if self.final_answer:
                raise ValueError("No main result returned by executor!")
            return None, execution_logs, False


__all__ = ["E2BExecutor"]
