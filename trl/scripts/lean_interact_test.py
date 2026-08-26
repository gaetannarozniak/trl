from lean_interact import LeanREPLConfig, LeanServer, Command, LocalProject

WS = "/storage/home/gaetan/fort-verl/services/lean/workspace"

config = LeanREPLConfig(
    project=LocalProject(directory=WS),
    local_repl_path=f"{WS}/.lake/packages/repl",
    lake_path="/home/gaetan/.elan/bin/lake",
    build_repl=False,          # already built
    verbose=True,
)
server = LeanServer(config)
print("ok")

lean_code = """
theorem add_zero_test (n : Nat) : n + 0 = n := by
  simp
"""

result = server.run(Command(cmd=lean_code))

print(result)
