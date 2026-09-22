# gbnf.py
from llama_cpp import LlamaGrammar

AGENT_GBNF = (
r"""
root ::= "<tool_call>\n" json-object "\n</tool_call>"

json-object ::= "{" ws ( pair ( "," ws pair )* )? ws "}"
pair ::= string ws ":" ws value
string ::= "\"" ([^"\\] | "\\" [^\x00])* "\""
value ::= string | number | json-object | array | "true" | "false" | "null"
array ::= "[" ws ( value ( "," ws value )* )? ws "]"
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= [ \t\n\r]*
"""
)

agent_grammar = LlamaGrammar.from_string(AGENT_GBNF)