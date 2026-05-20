import vllm
import dinfer
print("✅ Smoke test 通过：两个包均可成功导入")
print(f"vLLM 版本: {vllm.__version__}")
print(f"dinfer 版本: {getattr(dinfer, '__version__', '未知')}")
