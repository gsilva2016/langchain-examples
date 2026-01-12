from fastmcp import Client
import asyncio

config = {
        "mcpServers": {
            "hello": {
                "url": "http://127.0.0.1:8001/mcp/",
                "transport": "http"
            }
        }
    }

def pretty_print_tools(tools):
    print("\nAvailable MCP Tools:")
    for tool in tools:
        print(f"Tool: {tool.name}")
        
def parse_response(response):
    for idx, i in enumerate(response):
        print(f"Response {idx}:", i.text)
    print("-" * 50)
        
async def main():
    async with Client(config) as client:
        await client.ping()
        print("Server is reachable")
        
        tools = await client.list_tools()
        pretty_print_tools(tools)
        
        print("-" * 40)
        print("Calling tool: list_collections")
        response = await client.call_tool(
            name="list_collections",
        )
        parse_response(response.content)
        
        print("Calling tool: create_collection")
        response = await client.call_tool(
            name="create_collection",
            arguments={"collection_name": "sk", "dimension": 128, "overwrite": False},
        )
        parse_response(response.content)
    
        print("Calling tool: list_collections")
        response = await client.call_tool(
            name="list_collections",
        )
        parse_response(response.content)
        
asyncio.run(main())