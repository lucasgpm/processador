import { 
    pipeline, 
    env, 
    AutoTokenizer, 
    AutoConfig 
} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

// Bloqueio de rede automático da lib
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
        // 1. Pega o binário do modelo
        const modelBuffer = await reconstruirCerebroIA();
        const modeloPath = `${BASE_URL}meu-modelo/`;

        console.log("📂 Carregando arquivos de configuração...");
        const [configRes, tokenizerRes, tokenizerConfigRes] = await Promise.all([
            fetch(`${modeloPath}config.json`),
            fetch(`${modeloPath}tokenizer.json`),
            fetch(`${modeloPath}tokenizer_config.json`)
        ]);

        const configJSON = await configRes.json();
        const tokenizerJSON = await tokenizerRes.json();
        const tokenizerConfigJSON = await tokenizerConfigRes.json();

        // 2. Criamos a configuração e o tokenizer manualmente
        const config = new AutoConfig(configJSON);
        const tokenizer = new AutoTokenizer(tokenizerConfigJSON, tokenizerJSON);
        
        console.log("🔄 Montando Pipeline de forma direta...");
        
        /**
         * AQUI ESTÁ A MUDANÇA CRÍTICA:
         * Em vez de chamar Model.from_pretrained, passamos o buffer diretamente para o pipeline.
         * Para o pipeline aceitar o buffer sem tentar dar fetch, passamos 'null' ou um nome falso
         * no primeiro parâmetro e injetamos o modelo via opções.
         */
        classificador = await pipeline('text-classification', null, {
            model_data: modelBuffer, // O binário do ONNX
            config: config,          // O objeto de configuração
            tokenizer: tokenizer,    // O tokenizer já instanciado
            quantized: true,
            local_files_only: true
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
