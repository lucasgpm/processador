import { pipeline, env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

// Desligamos TUDO que for download automático da biblioteca
env.allowRemoteModels = false;
env.allowLocalModels = false;

let classificador;
let processarLinhasComClassificador;

async function reconstruirCerebroIA() {
    console.log("🧠 Baixando pedaços do modelo...");
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    const buffers = await Promise.all(partes.map(async (nome) => {
        const res = await fetch(path + nome);
        if (!res.ok) throw new Error(`Erro ao baixar parte: ${nome}`);
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
        // 1. Baixamos o binário (cérebro)
        const modelBuffer = await reconstruirCerebroIA();

        console.log("📂 Baixando arquivos de configuração manualmente...");
        const modeloPath = `${BASE_URL}meu-modelo/`;

        // 2. BAIXAMOS OS JSONS NÓS MESMOS (Sem passar pela lib)
        const [configRes, tokenizerRes, tokenizerConfigRes] = await Promise.all([
            fetch(`${modeloPath}config.json`),
            fetch(`${modeloPath}tokenizer.json`),
            fetch(`${modeloPath}tokenizer_config.json`)
        ]);

        if (!configRes.ok || !tokenizerRes.ok || !tokenizerConfigRes.ok) {
            throw new Error("Erro ao baixar arquivos JSON do seu GitHub.");
        }

        const configData = await configRes.json();
        const tokenizerData = await tokenizerRes.json();
        const tokenizerConfigData = await tokenizerConfigRes.json();

        console.log("🔄 Inicializando Tokenizer com dados locais...");
        
        // 3. Criamos o tokenizer passando os dados já baixados
        const tokenizer = new AutoTokenizer(tokenizerConfigData, tokenizerData);

        console.log("🚀 Montando Pipeline final...");

        // 4. Montamos o pipeline injetando TUDO manualmente
        classificador = await pipeline('text-classification', modelBuffer, {
            tokenizer: tokenizer,
            config: configData,
            quantized: true
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
        console.error("❌ Erro fatal no Worker:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
