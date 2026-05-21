"""Epoch-Spark configuration constants."""

MODEL_PATH = "/mnt/models/LLaDA2.0-mini"

# LLaDA2 MoE architecture
NUM_LAYERS = 20
DENSE_LAYERS = [0]  # layer 0 is dense FFN
MOE_LAYERS = list(range(1, 20))  # layers 1-19 are MoE
NUM_EXPERTS = 256
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 512
ROUTED_SCALING_FACTOR = 2.5
VOCAB_SIZE = 157184

# Token IDs
MASK_ID = 156895
PAD_ID = 156892
EOS_ID = 156892

# Diffusion generation
BLOCK_LENGTH = 32
DEFAULT_GEN_LENGTH = 256
DEFAULT_STEPS_PER_BLOCK = 12

# Neuron tile configuration
NEURON_TILE_SIZE = 64  # neurons per tile within each expert
TILES_PER_EXPERT = MOE_INTERMEDIATE_SIZE // NEURON_TILE_SIZE  # 512/64 = 8

# Weight sizes (bf16 = 2 bytes)
BYTES_PER_PARAM = 2
EXPERT_WEIGHT_BYTES = (
    MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE * BYTES_PER_PARAM * 3  # gate + up + down
)  # 512*2048*2*3 = 6,291,456 ~ 6.3MB per expert
LAYER_EXPERTS_BYTES = EXPERT_WEIGHT_BYTES * NUM_EXPERTS  # ~1.6GB per MoE layer

# Residency manager defaults
DEFAULT_GPU_EXPERT_BUDGET = 80  # experts per layer in GPU cache
DEFAULT_DECODED_CACHE_REFRESH_M = 5  # refresh decoded-token cache every M iterations
DEFAULT_DFR_DECAY = 0.67  # EMA decay for expert heat scores (from SparkInfer)

# Benchmark prompts
PROMPTS = [
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
    "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
    "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
    "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
]
