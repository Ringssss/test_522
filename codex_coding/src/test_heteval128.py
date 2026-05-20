#!/usr/bin/env python3
"""
v0.1.15.8k — HetEval-128: C5 vs C10-M∞ at batch=128

Hypothesis: larger batch → more unique active experts without S_mask →
EB's S_mask restriction saves more → C10 may beat C5.

128 heterogeneous prompts, gen_length=256.
"""

from __future__ import annotations
import os, sys, time, socket, json
from pathlib import Path
from collections import OrderedDict

import torch

REPO_ROOT = Path("/home/wuhang/wuhang/dllm_wh")
sys.path.insert(0, str(REPO_ROOT / "codex_coding" / "src"))

from test_fused_eb_triton import (
    fused_routing, FusedEBController,
    _kernel_A, _kernel_A_cold, _kernel_B_v2, _kernel_B_v3,
    _kernel_C, _kernel_D_v2, _kernel_zero_init,
)
from baseline_optimizations import apply_all_optimizations

MASK_ID, EOS_ID = 156895, 156892
BLOCK_LENGTH = 32  # default, overridden by --block-length
MODEL_PATH = "/mnt/models/LLaDA2.0-mini"
DEVICE = "cuda:0"
GEN_LENGTH = 256
BATCH_SIZE = 128

# 128 heterogeneous prompts (first 32 = original HetEval-32)
PROMPTS = [
    # === Original 32 (HetEval-32) ===
    "Please solve the following problems step by step.\n\nProblem 1: A train travels from City A to City B at 80 km/h and returns at 60 km/h. The total distance between the two cities is 240 km. What is the average speed for the entire round trip?\n\nProblem 2: A rectangular garden has a perimeter of 56 meters.",
    "Write a detailed essay about the history of artificial intelligence, covering the Dartmouth conference of 1956, the AI winters, the rise of machine learning in the 1990s, and deep learning breakthroughs.",
    "You are a chemistry professor. Explain Le Chatelier's principle with examples and how it applies to industrial ammonia production via the Haber process.",
    "Design a complete REST API for an e-commerce platform with endpoints for user authentication, product management, shopping cart operations, and order processing.",
    "Analyze the global economic impact of climate change across agriculture, energy, real estate, and healthcare sectors with specific examples.",
    "Explain quantum computing to a classical CS background: qubits, superposition, entanglement, Shor's algorithm, and current hardware approaches.",
    "You are a systems architect. Design a distributed message queue with partition-based storage, consumer groups, replication, and exactly-once semantics.",
    "Write a comprehensive guide to training large language models covering data collection, tokenizer training, architecture decisions, and distributed training strategies.",
    "Solve the quadratic equation x^2 - 5x + 6 = 0 step by step. Show the factoring method, then verify both solutions by substituting them back into the original equation.",
    "Explain the mathematical foundations of neural networks: backpropagation, gradient descent, loss functions, and the universal approximation theorem.",
    "Design a microservices architecture for a ride-sharing application with real-time matching, pricing, routing, payments, and driver management.",
    "Write about the history of cryptography from Caesar ciphers through RSA, elliptic curve cryptography, and post-quantum cryptographic algorithms.",
    "Explain database indexing strategies: B-trees, hash indexes, bitmap indexes, and their trade-offs for OLTP vs OLAP workloads.",
    "Solve this logic puzzle step by step: If A is true, then B is true. If B is true, then C is true. A is true. What can we conclude about B and C? Then, if D is true only when both B and C are true, what can we conclude about D?",
    "Design a CI/CD pipeline for a large monorepo with microservices, including build caching, parallel testing, canary deployments, and rollback strategies.",
    "Explain the theory of relativity to a physics undergraduate, covering special relativity, time dilation, length contraction, and general relativity basics.",
    "Write a comprehensive comparison of Python, Rust, and Go for systems programming, covering memory safety, concurrency models, and ecosystem maturity.",
    "Design a real-time recommendation engine for a video streaming platform that handles cold start, user preferences, and content diversity.",
    "Explain the CAP theorem and its practical implications for distributed database design, with examples from Cassandra, MongoDB, and CockroachDB.",
    "Write a Python function to compute the nth Fibonacci number. Show the function, then compute fib(1) through fib(10) step by step and list all 10 values.",
    "Design a fraud detection system for a payment processing company using machine learning, rule engines, and real-time streaming analytics.",
    "Explain compiler optimization techniques including SSA form, loop unrolling, vectorization, and register allocation strategies.",
    "Write about the history and future of space exploration, from Apollo missions through SpaceX reusability to planned Mars colonization.",
    "Design an observability platform with distributed tracing, log aggregation, metrics collection, and intelligent alerting for microservices.",
    "Explain the mathematics behind public key cryptography, including modular arithmetic, Euler's theorem, and the RSA algorithm step by step.",
    "Write a guide to modern CSS layout techniques including Flexbox, Grid, Container Queries, and responsive design best practices.",
    "Design a multi-tenant SaaS platform architecture with data isolation, custom domains, billing integration, and horizontal scaling.",
    "Explain how garbage collectors work in JVM, Go, and Python, comparing mark-sweep, generational, and reference counting approaches.",
    "List all 8 planets in our solar system in order from closest to farthest from the Sun. For each planet, state whether it is a terrestrial or gas/ice giant planet, and give its approximate orbital period in Earth years.",
    "Design a real-time collaborative document editor like Google Docs with conflict resolution, offline support, and version history.",
    "Explain operating system memory management: virtual memory, page tables, TLB, demand paging, and memory-mapped files.",
    "Write a comprehensive guide to Kubernetes architecture including pods, services, ingress, operators, and cluster autoscaling.",
    # === Additional 96 prompts ===
    "Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions, and explain the role of chlorophyll.",
    "Design a banking system's database schema with accounts, transactions, loans, and audit trails, considering ACID compliance.",
    "Explain the differences between TCP and UDP protocols, their use cases, and how congestion control works in TCP.",
    "Write a tutorial on Git branching strategies: GitFlow, trunk-based development, and feature flags for continuous delivery.",
    "Analyze the ethical implications of facial recognition technology in public surveillance and law enforcement.",
    "Explain how transformers work in deep learning, covering self-attention, multi-head attention, positional encoding, and the encoder-decoder architecture.",
    "Design a smart home automation system with IoT devices, edge computing, cloud integration, and voice control.",
    "Write about the evolution of programming languages from Fortran to modern languages like Kotlin, Swift, and Zig.",
    "Explain the principles of thermodynamics: the four laws, entropy, enthalpy, and their applications in engineering.",
    "Design a content delivery network (CDN) with edge caching, origin shielding, cache invalidation, and DDoS protection.",
    "Write a comparative analysis of democratic systems: presidential vs parliamentary, proportional vs first-past-the-post.",
    "Explain how CRISPR-Cas9 gene editing works, its applications in medicine and agriculture, and ethical considerations.",
    "Design a real-time stock trading platform with order matching, market data feeds, risk management, and regulatory compliance.",
    "Write about the Renaissance period: its origins in Italy, key figures like Leonardo da Vinci and Michelangelo, and its lasting impact on art and science.",
    "Explain the mechanics of black holes: formation, event horizon, Hawking radiation, and information paradox.",
    "Design a machine learning pipeline for image classification with data augmentation, transfer learning, and model serving.",
    "Write a guide to functional programming concepts: pure functions, immutability, higher-order functions, monads, and pattern matching.",
    "Explain how vaccines work: types of vaccines, immune response mechanisms, herd immunity, and mRNA technology.",
    "Design an autonomous vehicle's perception system with lidar, radar, camera fusion, and real-time object detection.",
    "Write about the Industrial Revolution: causes, key inventions, social impacts, and the transition from agrarian to industrial economies.",
    "Explain the fundamentals of signal processing: Fourier transforms, sampling theorem, filtering, and applications in audio processing.",
    "Design a distributed file storage system like HDFS with replication, erasure coding, metadata management, and fault tolerance.",
    "Write a comprehensive overview of the human immune system: innate vs adaptive immunity, T cells, B cells, and antibodies.",
    "Explain graph algorithms: BFS, DFS, Dijkstra's algorithm, minimum spanning trees, and their applications in network analysis.",
    "Design a natural language processing pipeline for sentiment analysis with tokenization, embeddings, and fine-tuning.",
    "Write about ancient Roman engineering: aqueducts, roads, concrete, and the architectural innovations of the Colosseum.",
    "Explain the economics of supply and demand, market equilibrium, elasticity, and government interventions like price controls.",
    "Design a serverless architecture on AWS with Lambda, API Gateway, DynamoDB, Step Functions, and event-driven patterns.",
    "Write about the history of the Internet: ARPANET, TCP/IP, the World Wide Web, and the evolution to Web 3.0.",
    "Explain organic chemistry basics: functional groups, reaction mechanisms, stereochemistry, and naming conventions.",
    "Design a music streaming service backend with catalog management, recommendation engine, offline sync, and royalty tracking.",
    "Write about climate science: greenhouse effect, carbon cycle, ocean acidification, and geoengineering proposals.",
    "Explain how operating system schedulers work: round-robin, priority-based, CFS, and real-time scheduling algorithms.",
    "Design an election voting system with voter verification, ballot secrecy, auditability, and protection against manipulation.",
    "Write about the psychology of decision-making: cognitive biases, heuristics, prospect theory, and nudge theory.",
    "Explain wireless networking: Wi-Fi 6, 5G NR, Bluetooth Low Energy, and the physical layer fundamentals.",
    "Design a hospital management system with patient records, appointment scheduling, billing, and inventory management.",
    "Write about the philosophy of science: falsifiability, paradigm shifts, the demarcation problem, and scientific realism.",
    "Explain containerization technology: Docker internals, cgroups, namespaces, overlay filesystems, and OCI standards.",
    "Design a social media platform's news feed algorithm with relevance ranking, content diversity, and abuse detection.",
    "Write about the history of mathematics: from Babylonian number systems through calculus to modern abstract algebra.",
    "Explain the principles of aerodynamics: lift, drag, Bernoulli's principle, and how airplane wings generate flight.",
    "Design a warehouse management system with inventory tracking, pick-and-pack optimization, and robotic integration.",
    "Write about marine biology: coral reef ecosystems, deep-sea organisms, ocean currents, and marine conservation.",
    "Explain the architecture of modern CPUs: pipelining, out-of-order execution, branch prediction, and cache hierarchy.",
    "Design a language learning app with spaced repetition, speech recognition, gamification, and adaptive difficulty.",
    "Write about the Silk Road: its historical routes, cultural exchanges, trade goods, and impact on civilizations.",
    "Explain probability theory: Bayes' theorem, conditional probability, random variables, and the central limit theorem.",
    "Design a disaster recovery system with RPO/RTO targets, failover automation, data replication, and runbook automation.",
    "Write about the development of electric vehicles: battery technology, charging infrastructure, and environmental impact.",
    "Explain the basics of quantum mechanics: wave-particle duality, uncertainty principle, Schrodinger equation, and quantum tunneling.",
    "Design a food delivery platform with restaurant management, driver routing, order tracking, and dynamic pricing.",
    "Write about ancient Greek philosophy: Socrates, Plato, Aristotle, and their influence on Western thought.",
    "Explain how search engines work: web crawling, indexing, PageRank algorithm, and query processing.",
    "Design a video conferencing system with low-latency streaming, screen sharing, breakout rooms, and recording.",
    "Write about renewable energy sources: solar, wind, hydroelectric, geothermal, and their grid integration challenges.",
    "Explain the principles of color theory: color models (RGB, CMYK, HSL), color harmony, and perception psychology.",
    "Design a customer relationship management (CRM) system with lead tracking, pipeline management, and analytics.",
    "Write about the French Revolution: causes, key events, the Reign of Terror, and its impact on modern democracy.",
    "Explain tensor operations in deep learning: broadcasting, einsum, reshaping, and their GPU implementation.",
    "Design a ride-pooling algorithm that matches multiple passengers going in similar directions to share rides.",
    "Write about the biology of aging: telomeres, cellular senescence, caloric restriction, and longevity research.",
    "Explain network security: firewalls, intrusion detection systems, TLS/SSL, and zero-trust architecture.",
    "Design a digital library system with full-text search, citation management, DOI resolution, and access control.",
    "Write about behavioral economics: loss aversion, anchoring, mental accounting, and their marketing applications.",
    "Explain the fundamentals of control systems: PID controllers, feedback loops, stability analysis, and Bode plots.",
    "Design a sports analytics platform for player performance tracking, game strategy analysis, and injury prediction.",
    "Write about the history of vaccines: from smallpox inoculation to modern mRNA vaccines and global immunization.",
    "Explain distributed consensus algorithms: Paxos, Raft, PBFT, and their trade-offs in fault tolerance and performance.",
    "Design an augmented reality navigation app with indoor positioning, 3D mapping, and real-time route guidance.",
    "Write about the golden age of Islamic science: contributions to mathematics, astronomy, medicine, and optics.",
    "Explain the principles of sound: acoustics, resonance, harmonics, and how musical instruments produce different tones.",
    "Design a carbon footprint tracking app with activity logging, emission calculations, and offset recommendations.",
    "Write about the space race: Sputnik, Apollo program, Moon landing, and its political and scientific significance.",
    "Explain how recommendation systems work: collaborative filtering, content-based filtering, and hybrid approaches.",
    "Design a peer-to-peer file sharing protocol with DHT-based discovery, chunk-based transfer, and integrity verification.",
    "Write about the neuroscience of memory: encoding, consolidation, retrieval, hippocampus function, and amnesia.",
    "Explain the fundamentals of robotics: kinematics, dynamics, path planning, and sensor integration.",
    "Design a subscription billing system with usage metering, invoicing, dunning management, and revenue recognition.",
    "Write about the Amazon rainforest: biodiversity, indigenous peoples, deforestation threats, and conservation efforts.",
    "Explain software testing methodologies: unit testing, integration testing, property-based testing, and mutation testing.",
    "Design a smart grid system for electricity distribution with demand response, renewable integration, and outage management.",
    "Write about the history of cinema: from silent films through talkies, Technicolor, CGI, to streaming platforms.",
    "Explain the mathematics of optimization: gradient descent, convex optimization, linear programming, and constraint satisfaction.",
    "Design a telemedicine platform with video consultations, prescription management, health records, and insurance integration.",
    "Write about plate tectonics: continental drift, seafloor spreading, subduction zones, and earthquake prediction.",
    "Explain microprocessor design: instruction set architecture, pipelining stages, hazards, and superscalar execution.",
    "Design a wildlife conservation tracking system with GPS collars, satellite imagery, population modeling, and poaching detection.",
    "Write about the evolution of money: from barter systems through coins, paper money, credit cards, to cryptocurrencies.",
    "Explain the principles of data compression: Huffman coding, LZ77, dictionary-based methods, and lossy vs lossless.",
    "Write about the geological history of Earth: formation, mass extinctions, ice ages, and the evidence from fossils.",
    "Explain how blockchain consensus mechanisms work: Proof of Work, Proof of Stake, and Byzantine fault tolerance.",
    "Design a precision agriculture system with soil sensors, drone imaging, weather integration, and yield optimization.",
    "Write about the cultural impact of jazz music: its African American origins, evolution through bebop to fusion.",
    "Explain the fundamentals of linear algebra: vector spaces, eigenvalues, matrix decompositions, and their applications in data science.",
    "Design an emergency response coordination system with real-time dispatch, resource allocation, and multi-agency communication.",
]

VERIFIABLE = {
    0: "average speed = 480/7 ~ 68.57 km/h", 8: "x = 2 and x = 3",
    13: "B is true, C is true, D is true", 19: "fib(10) = 55",
    28: "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune",
}

# M=inf controller (cold-only, no hot update)
class ColdOnlyEBController(FusedEBController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_init_history = []
        self.s_mask_cache = {}

    def cold_path(self, layer_idx, logits, bias):
        N, E = logits.shape
        b = self._get_bufs(N, E, logits.device)
        _kernel_zero_init[(1,)](b['pop'], b['sat_flag'], b['sat_count'],
                                b['G'], b['H'], E=E)
        lf = logits.float(); bf = bias.float()
        _kernel_A_cold[(N,)](lf, bf, b['pop'], b['topkm_idx'], b['topkm_w'], b['r'],
                             N, self.rsf, self.quality_floor,
                             lf.stride(0), lf.stride(1),
                             b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                             E=E, KEXT=self.K_ext, KEXT_PAD=16, K=self.K)
        _kernel_B_v3[(1,)](b['pop'], b['s_mask'], self.K_target, E=E)
        q_major_x1000 = int(self.q_major * 1000)
        for _ in range(self.MAX_ROUNDS):
            _kernel_C[(N,)](b['topkm_idx'], b['topkm_w'], b['r'],
                           b['s_mask'], b['sat_flag'], b['sat_count'], b['G'], b['H'],
                           N, b['topkm_idx'].stride(0), b['topkm_idx'].stride(1),
                           E=E, KEXT=self.K_ext, KEXT_PAD=16)
            _kernel_D_v2[(1,)](b['s_mask'], b['sat_flag'], b['sat_count'],
                              b['G'], b['H'], N, q_major_x1000, E=E, CAP=self.cap)
        actual_s = int(b['s_mask'].sum().item())
        self.K_init[layer_idx] = actual_s
        self.k_init_history.append(actual_s)
        if layer_idx not in self.s_mask_cache:
            self.s_mask_cache[layer_idx] = torch.zeros(E, device=logits.device, dtype=torch.int32)
        self.s_mask_cache[layer_idx].copy_(b['s_mask'])
        self.cold_count += 1
        return b['s_mask']

    def hot_path(self, layer_idx, logits, bias):
        self.hot_count += 1
        return self.s_mask_cache[layer_idx]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--block-length', type=int, default=32)
    args, _ = parser.parse_known_args()
    global BLOCK_LENGTH
    BLOCK_LENGTH = args.block_length

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)
    torch.cuda.set_device(device)

    from vllm import distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from dinfer import (BlockDiffusionLLM, BlockIteratorFactory, KVCacheFactory,
                        ThresholdParallelDecoder)
    from dinfer.model import LLaDA2MoeModelLM
    from transformers import AutoTokenizer, AutoConfig

    port = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port.bind(("127.0.0.1", 0)); p = port.getsockname()[1]; port.close()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(p)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, backend="nccl")

    print("=" * 80)
    print(f"HetEval-128: C5 vs C10-M∞ at batch={BATCH_SIZE}")
    print(f"  gen_length={GEN_LENGTH}, block_length={BLOCK_LENGTH}, q_major=1.0")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)

    pcfg = ParallelConfig(enable_expert_parallel=True)
    with set_current_vllm_config(VllmConfig(parallel_config=pcfg)):
        model = LLaDA2MoeModelLM(config=config).eval()
        model.load_weights(MODEL_PATH, torch_dtype=torch.bfloat16, device=device)
        model = model.to(device)
        with torch.inference_mode():
            _ = model(torch.arange(180, dtype=torch.long, device=device).unsqueeze(0),
                      use_cache=False)

        print("\nApplying baseline optimizations...")
        n_rms, n_fa = apply_all_optimizations(model)
        print(f"  Fused RMSNorm: {n_rms}, Flash-attn: {n_fa}")

        # Build input
        all_ids = []
        for i in range(BATCH_SIZE):
            text = PROMPTS[i]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    add_generation_prompt=True, tokenize=False)
            all_ids.append(tokenizer(text, return_tensors="pt")["input_ids"][0])
        mx = max(x.shape[0] for x in all_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [torch.cat([torch.full((mx - ids.shape[0],), pad_id, dtype=ids.dtype), ids])
                  if ids.shape[0] < mx else ids for ids in all_ids]
        input_ids = torch.stack(padded, dim=0).to(device)
        prompt_len = input_ids.shape[1]
        print(f"  Input shape: {input_ids.shape} (batch={BATCH_SIZE})")
        print(f"  GPU memory: {torch.cuda.memory_allocated(device)/1e9:.1f}GB / "
              f"{torch.cuda.get_device_properties(device).total_memory/1e9:.1f}GB")

        decoder_t0 = ThresholdParallelDecoder(
            temperature=0.0, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)
        decoder_t7 = ThresholdParallelDecoder(
            temperature=0.7, threshold=0.90, mask_id=MASK_ID, eos_id=EOS_ID)

        def make_dllm(decoder):
            return BlockDiffusionLLM(
                model, decoder,
                BlockIteratorFactory(use_block_diffusion=True),
                cache_factory=KVCacheFactory("prefix", is_bd_model=True),
                early_stop=True, maximum_unroll=4, expected_tpf=15,
                backend='vllm', lazy_cache_update=True,
                inplace_cache_update=True)

        orig_routings = {}
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "LLaDA2MoeGate":
                orig_routings[name] = mod.routing

        def restore():
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate" and name in orig_routings:
                    mod.routing = orig_routings[name]

        def patch_c5():
            restore()
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    def mk(bb, rr, tt, nn, gg):
                        def fn(hs, go, topk, renorm):
                            w, i = fused_routing(go, bb, rr, s_mask=None, K=tt, ng=nn, tkg=gg)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg)

        def patch_eb(ctrl):
            restore()
            idx = 0
            for name, mod in model.named_modules():
                if mod.__class__.__name__ == "LLaDA2MoeGate":
                    b, r, tk, ng, tkg = (mod.expert_bias, mod.routed_scaling_factor,
                                         mod.top_k, mod.n_group, mod.topk_group)
                    li = idx
                    def mk(bb, rr, tt, nn, gg, layer_i, cc):
                        def fn(hs, go, topk, renorm):
                            sm = cc.get_s_mask(layer_i, go, bb)
                            w, i = fused_routing(go, bb, rr, s_mask=sm, K=tt, ng=nn, tkg=gg)
                            return w.to(go.dtype), i
                        return fn
                    mod.routing = mk(b, r, tk, ng, tkg, li, ctrl)
                    idx += 1

        results = OrderedDict()

        # ---- C5 ----
        print(f"\n{'='*60}")
        print("C5: Fused routing only")
        print(f"{'='*60}")
        patch_c5()
        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup done: {dllm.diff_iteration.num_forwards} fwd")

        c5_times, c5_fwds = [], []
        for ri in range(2):
            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            c5_times.append(t1-t0); c5_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd")
        results['C5'] = {'avg_time': sum(c5_times)/2, 'avg_fwd': sum(c5_fwds)/2,
                         'ms_per_fwd': sum(c5_times)/2 / (sum(c5_fwds)/2) * 1000}

        # ---- C10-M∞ ----
        print(f"\n{'='*60}")
        print("C10-M∞: cold-only EB (q_major=1.0)")
        print(f"{'='*60}")
        ctrl = ColdOnlyEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)
        patch_eb(ctrl)

        dllm = make_dllm(decoder_t0)
        with torch.inference_mode():
            dllm.diff_iteration.num_forwards = 0
            _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        print(f"  Warmup: {dllm.diff_iteration.num_forwards} fwd, "
              f"cold={ctrl.cold_count}, hot={ctrl.hot_count}")
        if ctrl.k_init_history:
            print(f"  |S| avg={sum(ctrl.k_init_history)/len(ctrl.k_init_history):.1f}, "
                  f"min={min(ctrl.k_init_history)}, max={max(ctrl.k_init_history)}")

        eb_times, eb_fwds = [], []
        for ri in range(2):
            ctrl.prev_N.clear(); ctrl.K_init.clear()
            ctrl.cold_count = 0; ctrl.hot_count = 0
            ctrl._bufs.clear(); ctrl.k_init_history.clear()
            ctrl.s_mask_cache.clear()

            dllm = make_dllm(decoder_t0); torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.inference_mode():
                dllm.diff_iteration.num_forwards = 0
                _ = dllm.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            eb_times.append(t1-t0); eb_fwds.append(dllm.diff_iteration.num_forwards)
            print(f"    Run {ri+1}: {t1-t0:.3f}s, {dllm.diff_iteration.num_forwards} fwd "
                  f"| cold={ctrl.cold_count} hot={ctrl.hot_count}")

        results['C10_Minf'] = {
            'avg_time': sum(eb_times)/2, 'avg_fwd': sum(eb_fwds)/2,
            'ms_per_fwd': sum(eb_times)/2 / (sum(eb_fwds)/2) * 1000,
            'cold': ctrl.cold_count, 'hot': ctrl.hot_count,
            'k_init_avg': sum(ctrl.k_init_history)/len(ctrl.k_init_history) if ctrl.k_init_history else 0,
            'k_init_min': min(ctrl.k_init_history) if ctrl.k_init_history else 0,
            'k_init_max': max(ctrl.k_init_history) if ctrl.k_init_history else 0,
        }

        # ---- Quality check C10-M∞ ----
        print(f"\n{'='*60}")
        print("Quality check — C10-M∞ (temp=0.7)")
        print(f"{'='*60}")
        restore()
        ctrl_q = ColdOnlyEBController(
            num_layers=19, K=8, M=4, K_target=40,
            quality_floor=0.70, q_major=1.0, per_round_cap=8)
        patch_eb(ctrl_q)
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            _ = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        torch.cuda.synchronize()
        ctrl_q.prev_N.clear(); ctrl_q.K_init.clear()
        ctrl_q.cold_count = 0; ctrl_q.hot_count = 0
        ctrl_q._bufs.clear(); ctrl_q.s_mask_cache.clear()
        dllm_q = make_dllm(decoder_t7)
        with torch.inference_mode():
            out = dllm_q.generate(input_ids.clone(), gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH)
        gen_tokens = out[:, prompt_len:]
        for bi in sorted(VERIFIABLE.keys()):
            gt = gen_tokens[bi]
            valid = gt[(gt != 0) & (gt != EOS_ID) & (gt != MASK_ID)]
            text = tokenizer.decode(valid, skip_special_tokens=True)
            print(f"  #{bi}: {text[:150]}")

        # ---- Summary ----
        print(f"\n{'='*80}")
        print(f"SUMMARY (batch={BATCH_SIZE}, block={BLOCK_LENGTH})")
        print(f"{'='*80}")
        c5_t = results['C5']['avg_time']
        c5_f = results['C5']['avg_fwd']
        print(f"  {'Config':<20s} {'Time(s)':>8s} {'Fwd':>5s} {'ms/fwd':>8s} {'vs C5':>8s} {'dFwd':>6s} {'|S| avg':>8s}")
        print(f"  {'-'*66}")
        for cn, r in results.items():
            d = (r['avg_time'] - c5_t) / c5_t * 100
            df = r['avg_fwd'] - c5_f
            ki = f"{r.get('k_init_avg', 0):.0f}" if 'k_init_avg' in r else "—"
            print(f"  {cn:<20s} {r['avg_time']:>8.3f} {r['avg_fwd']:>5.0f} "
                  f"{r['ms_per_fwd']:>8.2f} {d:>+7.1f}% {df:>+5.0f} {ki:>8s}")

        # Compare with batch=32 reference
        print(f"\n  Reference (batch=32): C5=7.69s/272fwd, C10-M∞=8.00s/274fwd (+4.0%)")

        out_path = REPO_ROOT / "codex_coding" / "results" / f"heteval128_blk{BLOCK_LENGTH}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
