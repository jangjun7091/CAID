# CAID
Computer AI Design: Accelerating Hardware Innovation through Agentic AI



# CAID: Computer AI Design 🚀
**Breaking the Bottleneck of Hardware Innovation through Agentic AI**

---

> "CAID는 하드웨어 설계의 패러다임을 바꾸기 위한 오픈 소스 프로젝트입니다. 기존의 CAD가 인간의 설계를 '보조'하는 도구였다면, CAID는 LLM과 World Model을 결합하여 AI가 직접 설계의 주체(Primary Designer)가 되는 세상을 꿈꿉니다."

---

### 🌟 Why CAID?
오늘날 소프트웨어의 발전 속도에 비해 하드웨어 혁신은 느립니다. 그 중심에는 '수동 모델링'이라는 병목 현상이 있습니다. 숙련된 엔지니어가 수백 시간을 들여 직접 도면을 그려야 하는 제약을 극복하기 위해, CAID는 **자연어(Text)만으로 3D 부품과 복잡한 어셈블리를 생성**하는 것을 목표로 합니다.

### 🛠️ Key Features
* **Generative Parametric Design:** 단순한 형상이 아닌, 수정 가능한 파라메트릭 코드(CadQuery, OpenSCAD 등)를 생성합니다.
* **Agentic Multi-Agent System:**
    * **Architect:** 설계 의도 분석 및 전체 전략 수립
    * **Designer:** 세부 부품 모델링 코드 생성
    * **Critic:** 제조 가능성(DFM) 및 물리적 타당성 검토
* **Physics-Aware Modeling:** 물리 법칙을 이해하고 해석(FEA) 결과를 설계에 스스로 반영합니다.

### 📂 Repository Structure
* `📂 agents/`: 설계, 비평, 조율을 담당하는 AI 에이전트 로직
* `📂 core/`: LLM 래퍼 및 물리적 상식을 반영한 월드 모델 엔진
* `📂 geometry/`: 오픈 소스 CAD 커널 연동 모듈
* `📂 sim/`: 자동화된 물리 해석 및 공차 검토 엔진
* `📂 examples/`: 2차 전지 모듈 설계 등 실전 적용 케이스

## 🗺️ CAID Roadmap: The Modular Synthesis Path

### ✅ Phase 1 & 2: Core & Assembly (Completed)
- [x] Natural language to 3D geometry (CadQuery)
- [x] Basic interference detection & self-correction
- [x] Initial 2-part assembly capability

### 🚀 Phase 3: Smart Library & Part Intelligence (Current)
- [ ] **Part Repository**: Saving generated models as reusable "Bricks".
- [ ] **Functional Metadata**: AI tagging of mounting points, holes, and axes.
- [ ] **Structural Sanity Check**: Light-weight feedback on model viability (Non-precise FEA).

### 🛠️ Phase 4: Advanced Modular Assembly (Planned)
- [ ] **Semantic Mating**: Text-based assembly commands (e.g., "Attach A to B via M3 holes").
- [ ] **Global Tolerance Control**: Automatic clearance adjustment for real-world assembly.
- [ ] **Standard Part Integration**: Using off-the-shelf parts (Bolts, Motors) in the assembly.

### 🌌 Phase 5: Autonomous System Building (Vision)
- [ ] **Recursive Synthesis**: Building complex machines from modular sub-assemblies.
- [ ] **Generative Layout**: AI proposing optimal part arrangements for a given goal.

---

### 🤝 How to Contribute
CAID는 인류의 제작 능력을 확장하고자 하는 모든 '호모 파베르(Homo Faber)'를 환영합니다. 이슈 제안, 코드 기여, 아이디어 공유 등 무엇이든 좋습니다.

### 📄 License
This project is licensed under the **MIT License**.
