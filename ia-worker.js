import { 
    pipeline, 
    env, 
    AutoTokenizer, 
    DistilBertForSequenceClassification // Classe específica, chega de "Auto"
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

env.allowRemoteModels = false;
env.allowLocalModels = true; 

let classificador;
let processarLinhasComClassificador;

async function reconstruirCerebroIA() {
    console.log("🧠 Baixando pedaços do modelo...");
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    const buffers = await Promise.all(partes.map(async (nome) => {
        const res = await fetch(path + nome);
        if (!res.ok) throw new Error(`Erro ao baixar: ${nome}`);
        return res.arrayBuffer();
    }));

    const totalLength = buffers.reduce((acc, b) => acc + b.byteLength, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const b of buffers) {
        combined.set(new Uint8Array(b), offset);
        offset += b.byteLength;
    }
    return combined;
}

const carregarIA = async () => {
    if (!classificador) {
        const modelBuffer = await reconstruirCerebroIA();
        const modeloPath = `${BASE_URL}meu-modelo/`;

        console.log("📂 Carregando JSONs...");
        const [configRes, tokenizerRes, tokenizerConfigRes] = await Promise.all([
            fetch(`${modeloPath}config.json`),
            fetch(`${modeloPath}tokenizer.json`),
            fetch(`${modeloPath}tokenizer_config.json`)
        ]);

        const configJSON = await configRes.json();
        const tokenizerJSON = await tokenizerRes.json();
        const tokenizerConfigJSON = await tokenizerConfigRes.json();

        console.log("🔄 Inicializando Tokenizer...");
        const tokenizer = new AutoTokenizer(tokenizerConfigJSON, tokenizerJSON);
        
        console.log("🔄 Forçando carregamento do DistilBert...");
        
        // AQUI ESTÁ A MUDANÇA: Usamos a classe específica e o método 'from_pretrained'
        // Mas o primeiro argumento é um nome qualquer, o que importa é o 'model_data'
        const model = await DistilBertForSequenceClassification.from_pretrained('distilbert', {
            model_data: modelBuffer,
            config: configJSON, // Passamos o JSON bruto direto
            quantized: true,
            local_files_only: true
        });

        console.log("🚀 Criando Pipeline...");
        
        // Injetamos o modelo já instanciado
        classificador = await pipeline('text-classification', model, {
            tokenizer: tokenizer
        });

        console.log("✅ IA Carregada com sucesso!");
        
        const modulo = await import(`${BASE_URL}processador.js`);
        processarLinhasComClassificador = modulo.processarLinhasComClassificador;
    }
    return classificador;
};

self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            if (tipo === 'PRELOAD') self.postMessage({ tipo: 'PRONTO' });
            if (tipo === 'PROCESSAR' && texto) {
                const dados = await processarLinhasComClassificador(texto.split('\n'), classificador);
                self.postMessage({ tipo: 'RESULTADO', dados });
            }
        }
    } catch (err) {
        console.error("❌ Erro no Worker:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
