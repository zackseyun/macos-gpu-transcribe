import Darwin
import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT

private struct ServerOptions {
    var model = "beshkenadze/cohere-transcribe-03-2026-mlx-4bit"
    var language: String? = "English"

    static func parse() -> ServerOptions {
        var options = ServerOptions()
        var args = CommandLine.arguments.dropFirst().makeIterator()
        while let arg = args.next() {
            switch arg {
            case "--model":
                if let value = args.next() { options.model = value }
            case "--language":
                if let value = args.next() { options.language = value }
            default:
                continue
            }
        }
        return options
    }
}

private struct Request: Decodable {
    let id: String?
    let audio: String
    let language: String?
    let maxTokens: Int?
    let chunkDuration: Float?
}

private struct Response: Encodable {
    let id: String?
    let ok: Bool
    let text: String?
    let error: String?
    let totalTime: Double?
    let wallTime: Double?
    let promptTokens: Int?
    let generationTokens: Int?
    let peakMemoryUsage: Double?
}

@main
enum CohereSTTServer {
    static func main() async {
        let options = ServerOptions.parse()
        do {
            let model = try await CohereTranscribeModel.fromPretrained(options.model)
            emit(["event": "ready", "model": options.model])
            while let line = readLine(strippingNewline: true) {
                guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { continue }
                autoreleasepool {
                    handle(line: line, model: model, defaultLanguage: options.language)
                }
            }
        } catch {
            emit(Response(id: nil, ok: false, text: nil, error: String(describing: error), totalTime: nil, wallTime: nil, promptTokens: nil, generationTokens: nil, peakMemoryUsage: nil))
            exit(1)
        }
    }

    private static func handle(line: String, model: CohereTranscribeModel, defaultLanguage: String?) {
        do {
            let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
            let url = URL(fileURLWithPath: (request.audio as NSString).expandingTildeInPath)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw NSError(domain: "CohereSTTServer", code: 2, userInfo: [NSLocalizedDescriptionKey: "Input audio file not found: \(url.path)"])
            }

            let (inputSampleRate, inputAudio) = try loadAudioArray(from: url)
            let audio = try prepareAudioForSTT(inputAudio, inputSampleRate: inputSampleRate, targetSampleRate: 16000)
            var params = model.defaultGenerationParameters
            params = STTGenerateParameters(
                maxTokens: request.maxTokens ?? 2048,
                temperature: params.temperature,
                topP: params.topP,
                topK: params.topK,
                verbose: false,
                language: normalizeLanguage(request.language ?? defaultLanguage),
                chunkDuration: request.chunkDuration ?? 30.0,
                minChunkDuration: params.minChunkDuration,
                repetitionPenalty: params.repetitionPenalty,
                repetitionContextSize: params.repetitionContextSize
            )

            let start = CFAbsoluteTimeGetCurrent()
            let output = model.generate(audio: audio, generationParameters: params)
            let wall = CFAbsoluteTimeGetCurrent() - start
            emit(Response(
                id: request.id,
                ok: true,
                text: output.text,
                error: nil,
                totalTime: output.totalTime,
                wallTime: wall,
                promptTokens: output.promptTokens,
                generationTokens: output.generationTokens,
                peakMemoryUsage: output.peakMemoryUsage
            ))
        } catch {
            let requestID = (try? JSONDecoder().decode(Request.self, from: Data(line.utf8)).id) ?? nil
            emit(Response(id: requestID, ok: false, text: nil, error: String(describing: error), totalTime: nil, wallTime: nil, promptTokens: nil, generationTokens: nil, peakMemoryUsage: nil))
        }
    }

    private static func prepareAudioForSTT(
        _ audio: MLXArray,
        inputSampleRate: Int,
        targetSampleRate: Int
    ) throws -> MLXArray {
        let mono = audio.ndim > 1 ? audio.mean(axis: -1) : audio
        guard inputSampleRate != targetSampleRate else { return mono }
        return try MLXAudioCore.resampleAudio(mono, from: inputSampleRate, to: targetSampleRate)
    }

    private static func normalizeLanguage(_ language: String?) -> String? {
        guard let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else { return nil }
        switch trimmed.lowercased() {
        case "en", "english": return "English"
        default: return trimmed
        }
    }

    private static func emit<T: Encodable>(_ value: T) {
        do {
            let data = try JSONEncoder().encode(value)
            if let line = String(data: data, encoding: .utf8) {
                print(line)
                fflush(stdout)
            }
        } catch {
            print("{\"ok\":false,\"error\":\"failed to encode response\"}")
            fflush(stdout)
        }
    }

    private static func emit(_ dict: [String: String]) {
        if let data = try? JSONSerialization.data(withJSONObject: dict), let line = String(data: data, encoding: .utf8) {
            print(line)
            fflush(stdout)
        }
    }
}
