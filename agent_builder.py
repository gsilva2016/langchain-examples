from langchain_core.tools import tool

@tool
def local_db_knowledge_base(query: str) -> int:
    """Provides the types of metrics provided by the OpenVINO Model Server"""
    return "Intel OpenVINO Model Server provides many metrics such as CPU and memory utilization. It also support ovms_infer_gabriel_time."

