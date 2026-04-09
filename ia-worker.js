import { 
    pipeline, 
    env, 
    AutoTokenizer, 
    DistilBertForSequenceClassification 
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

// Configurações de ambiente para isolamento total
env.allowRemoteModels = false;
env.allowLocalModels = true; 

let classificador;
let processarLinhasComClassificador;

/**
 * Reconstrói o modelo a partir dos pedaços binários
 */
async function reconstruirCerebroIA() {
    console.log("🧠 Baixando pedaços do DistilBERT...");
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

/**
 * Carregamento Ultra Master: Injeção Direta
 */
const carregarIA = async () => {
    if (!classificador) {
        // 1. Obtém o buffer binário
        const modelBuffer = await reconstruirCerebroIA();
        const modeloPath = `${BASE_URL}meu-modelo/`;

        console.log("📂 Carregando dicionários e configurações...");
        const [configRes, tokenizerRes, tokenizerConfigRes] = await Promise.all([
            fetch(`${modeloPath}config.json`),
            fetch(`${modeloPath}tokenizer.json`),
            fetch(`${modeloPath}tokenizer_config.json`)
        ]);

        const configData = await configRes.json();
        const tokenizerData = await tokenizerRes.json();
        const tokenizerConfigData = await tokenizerConfigRes.json();

        console.log("🔄 Inicializando Tokenizer Multilíngue...");
        const tokenizer = new AutoTokenizer(tokenizerConfigData, tokenizerData);
        
        console.log("🔄 Instanciando DistilBERT a partir do buffer...");
        
        /**
         * O SEGREDO DO MASTER:
         * Usamos a classe específica do seu modelo. 
         * Passamos um nome fictício 'distilbert', mas entregamos o model_data.
         */
        const model = await DistilBertForSequenceClassification.from_pretrained('distilbert', {
            model_data: modelBuffer,
            config: configData,
            quantized: true,
            local_files_only: true
        });

        console.log("🚀 Criando Pipeline de Classificação...");
        
        // Criamos o pipeline injetando o modelo e tokenizer prontos
        classificador = await pipeline('text-classification', model, {
            tokenizer: tokenizer
        });

        console.log("✅ IA Carregada com sucesso!");
        
        // Importa seu script de lógica
        const modulo = await import(`${BASE_URL}processador.js`);
        processarLinhasComClassificador = modulo.processarLinhasComClassificador;
    }
    return classificador;
};

/**
 * Escutador do Worker
 */
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            
            if (tipo === 'PRELOAD') {
                self.postMessage({ tipo: 'PRONTO' });
            }
            
            if (tipo === 'PROCESSAR' && texto) {
                if (typeof processarLinhasComClassificador === 'function') {
                    const dados = await processarLinhasComClassificador(texto.split('\n'), classificador);
                    self.postMessage({ tipo: 'RESULTADO', dados });
                } else {
                    throw new Error("Lógica de processamento não encontrada.");
                }
            }
        }
    } catch (err) {
        console.error("❌ Erro Master no Worker:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
