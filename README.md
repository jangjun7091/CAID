# CAID: Computer AI Design
**Breaking the Bottleneck of Hardware Innovation through Agentic AI**

---

> "CAID는 하드웨어 설계의 패러다임을 바꾸기 위한 오픈 소스 프로젝트입니다. 기존의 CAD가 인간의 설계를 '보조'하는 도구였다면, CAID는 LLM과 World Model을 결합하여 AI가 직접 설계의 주체(Primary Designer)가 되는 세상을 꿈꿉니다."

---

### Why CAID?
오늘날 소프트웨어의 발전 속도에 비해 하드웨어 혁신은 느립니다. 그 중심에는 '수동 모델링'이라는 병목 현상이 있습니다. 숙련된 엔지니어가 수백 시간을 들여 직접 도면을 그려야 하는 제약을 극복하기 위해, CAID는 **자연어(Text)만으로 3D 부품과 복잡한 어셈블리를 생성**하는 것을 목표로 합니다.

### Key Features
* **Generative Parametric Design:** 단순한 형상이 아닌, 수정 가능한 파라메트릭 코드(CadQuery)를 생성합니다.
* **Agentic Multi-Agent System:**
    * **Architect** (`agents/architect.py`): 설계 의도 분석 및 전체 전략 수립
    * **Designer** (`agents/designer.py`): 세부 부품 모델링 코드 생성
    * **Critic** (`agents/critic.py`): 제조 가능성(DFM) 및 물리적 타당성 검토
* **Physics-Aware Modeling:** FEA 결과(`sim/fea_engine.py`)와 공차 해석(`sim/tolerance.py`)을 설계에 자동 반영합니다.
* **REST API:** FastAPI 기반 엔드포인트(`api/routes.py`)로 외부 툴 연동이 가능합니다.

### Repository Structure
```
agents/          AI 에이전트 (Architect, Designer, Critic)
api/             FastAPI REST 엔드포인트
core/            LLM 래퍼, 오케스트레이터, 세션 관리, 스키마
geometry/        CadQuery / OpenSCAD 커널 연동
prompts/         Jinja2 프롬프트 템플릿
sim/             FEA, 공차(Tolerance), 시뮬레이션 서비스
examples/        실전 적용 케이스
output/          생성된 모델 파일 출력 디렉터리
```

---

## CAID Roadmap: The Modular Synthesis Path

### ✅ Phase 1 & 2: Core & Assembly (Completed)
- [x] Natural language → parametric 3D geometry (CadQuery)
- [x] Agentic self-correction loop with DFM critique (up to 3 iterations)
- [x] Lightweight FEA and tolerance stack-up analysis (`sim/`)
- [x] 2-part assembly with interference detection
- [x] FastAPI REST interface (`api/routes.py`)
- [x] CLI entrypoint (`caid_cli.py`)

---

### 🚀 Phase 3: Smart Part Library (Current Focus)

**Goal:** Turn every generated model into a reusable "Brick" that future agents can retrieve, compose, and adapt — eliminating redundant generation and enabling cumulative design knowledge.

| Task | Module | Description |
|------|--------|-------------|
| **Part Repository** | `library/repository.py` | SQLite-backed store: save/load/query parts by name, tags, parameters |
| **Functional Metadata** | `library/metadata.py` | AI-extracted annotations: mounting holes, mating axes, bounding box, material |
| **Semantic Search** | `library/search.py` | Embedding-based similarity search over part descriptions |
| **Structural Sanity Check** | `sim/quick_check.py` | Rule-based pre-FEA viability filter (wall thickness, overhang angle) |
| **Library API** | `api/routes.py` | `GET /parts`, `GET /parts/{id}`, `POST /parts` REST endpoints |

**First Task → See "Getting Started: Phase 3" below.**

---

### 🛠️ Phase 4: Advanced Modular Assembly (Planned)

**Goal:** Assemble multi-part systems from natural language commands, with automatic constraint resolution.

- **Semantic Mating:** Text commands like *"Attach A to B via M3 holes"* → constraint solver maps named features
- **Global Tolerance Control:** Automatic clearance/interference adjustment across the full assembly tree
- **Standard Part Integration:** Off-the-shelf catalogue (bolts, bearings, motors) pulled from the part library
- **Assembly Graph:** DAG representation of part relationships enabling recursive sub-assembly reuse

---

### 🌌 Phase 5: Autonomous System Building (Vision)

**Goal:** Given a high-level functional spec, CAID decomposes, designs, simulates, and validates a complete mechanical system with minimal human intervention.

- **Recursive Synthesis:** Complex machines built from modular sub-assemblies retrieved from the Phase 3 library
- **Generative Layout:** AI proposes optimal part arrangements to satisfy spatial, load, and thermal constraints
- **World Model Integration** (`core/world_model.py`): Physics priors guide the generative loop end-to-end
- **Closed-Loop Validation:** Simulation results automatically trigger re-design until specs are met

---

## Getting Started: Phase 3 — Part Library

The first concrete task for Phase 3 is implementing `library/repository.py`: a lightweight, queryable store for generated parts.

### What to build

```python
# library/repository.py
@dataclass
class PartRecord:
    id: str                  # UUID
    name: str
    description: str
    tags: list[str]
    parameters: dict         # e.g. {"width": 40, "height": 20, "material": "PLA"}
    cadquery_code: str        # the generated script
    stl_path: str | None     # path under output/
    created_at: str          # ISO-8601

class PartRepository:
    def save(self, part: PartRecord) -> str: ...
    def get(self, part_id: str) -> PartRecord | None: ...
    def search(self, query: str, tags: list[str] = []) -> list[PartRecord]: ...
    def list_all(self) -> list[PartRecord]: ...
```

### Integration points
1. **`core/orchestrator.py`** — call `repo.save()` after every successful generation
2. **`api/routes.py`** — expose `GET /parts` and `POST /parts` using `PartRepository`
3. **`agents/architect.py`** — call `repo.search()` before generating; reuse or adapt existing parts

### Suggested storage
- **SQLite** via `sqlite3` (stdlib, no extra dependency): one table `parts`, JSON columns for `tags` and `parameters`
- STL files remain under `output/`; the DB stores the path reference

---

### How to Contribute
CAID는 인류의 제작 능력을 확장하고자 하는 모든 '호모 파베르(Homo Faber)'를 환영합니다. 이슈 제안, 코드 기여, 아이디어 공유 등 무엇이든 좋습니다.

### License
This project is licensed under the **MIT License**.
