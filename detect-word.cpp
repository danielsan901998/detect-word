#include "whisper.h"
#include "common-whisper.h"

extern "C" {
#include <libavutil/log.h>
}

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

std::string clean_word(const std::string & word) {
    std::string cleaned;
    cleaned.reserve(word.length());
    for (unsigned char c : word) {
        if (std::isalnum(c)) {
            cleaned += (char)std::tolower(c);
        }
    }
    return cleaned;
}

void append_cleaned_word(const char * token_text, std::string & accumulated, std::vector<int> & char_to_token, int token_index) {
    if (!token_text) return;
    for (size_t i = 0; token_text[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)token_text[i];
        if (std::isalnum(c)) {
            accumulated += (char)std::tolower(c);
            char_to_token.push_back(token_index);
        }
    }
}

void whisper_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    static ggml_log_level last_level = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) {
        last_level = level;
    }
    if (last_level == GGML_LOG_LEVEL_ERROR || last_level == GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

int main(int argc, char ** argv) {
    whisper_log_set(whisper_log_callback, nullptr);
    av_log_set_level(AV_LOG_ERROR);

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <audio_file> <word> [--output <output_file>] [--model <path>] [--vad-model <path>] [--threads <n>] [--beam-size <n>]\n", argv[0]);
        return 1;
    }

    std::string audio_file = argv[1];
    std::string target_word = clean_word(argv[2]);
    std::string output_file = "/tmp/trim-output.opus";
    std::string model_path = "/home/daniel/archivos/ggml-large-v3-turbo-q5_0.bin";
    std::string vad_model_path = "/home/daniel/archivos/ggml-silero-v6.2.0.bin";
    int n_threads = std::thread::hardware_concurrency();
    int beam_size = 5;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--vad-model" && i + 1 < argc) {
            vad_model_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if (arg == "--beam-size" && i + 1 < argc) {
            beam_size = std::stoi(argv[++i]);
        }
    }

    // Load audio data
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(audio_file, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "Error: Failed to read audio data from %s\n", audio_file.c_str());
        return 1;
    }

    // Initialize VAD context
    struct whisper_vad_context_params vparams = whisper_vad_default_context_params();
    vparams.n_threads = n_threads;
    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(vad_model_path.c_str(), vparams);
    if (vctx == nullptr) {
        fprintf(stderr, "Error: Failed to initialize VAD context from %s\n", vad_model_path.c_str());
        return 1;
    }

    // Initialize whisper context
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "Error: Failed to initialize whisper context from %s\n", model_path.c_str());
        whisper_vad_free(vctx);
        return 1;
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    params.beam_search.beam_size = beam_size;
    params.print_progress = false;
    params.print_special = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.translate = false;
    params.language = "auto";
    params.n_threads = n_threads;
    params.token_timestamps = true;
    params.no_context = true;
    params.single_segment = false;
    params.suppress_blank = true;
    params.suppress_nst = true;

    float final_start_seconds = -1.0f;

    std::string accumulated_cleaned;
    std::vector<int> char_to_token;
    accumulated_cleaned.reserve(256);
    char_to_token.reserve(256);

    // Detect speech segments and process in 30s chunks
    struct whisper_vad_params vad_params = whisper_vad_default_params();
    const int chunk_size_samples = 30 * WHISPER_SAMPLE_RATE;

    for (int i = 0; i < (int)pcmf32.size(); i += chunk_size_samples) {
        int n_samples = std::min(chunk_size_samples, (int)pcmf32.size() - i);
        struct whisper_vad_segments * segments = whisper_vad_segments_from_samples(vctx, vad_params, pcmf32.data() + i, n_samples);
        if (segments == nullptr) {
            continue;
        }

        int n_vad_segments = whisper_vad_segments_n_segments(segments);
        for (int j = 0; j < n_vad_segments; ++j) {
            float t0_local = whisper_vad_segments_get_segment_t0(segments, j) * 0.01f;
            float t1_local = whisper_vad_segments_get_segment_t1(segments, j) * 0.01f;
            
            float t0 = (float)i / (float)WHISPER_SAMPLE_RATE + t0_local;
            float t1 = (float)i / (float)WHISPER_SAMPLE_RATE + t1_local;

            int sample_start = (int)(t0 * WHISPER_SAMPLE_RATE);
            int sample_count = (int)((t1 - t0) * WHISPER_SAMPLE_RATE);

            if (sample_start >= (int)pcmf32.size()) continue;
            if (sample_start + sample_count > (int)pcmf32.size()) {
                sample_count = (int)pcmf32.size() - sample_start;
            }
            if (sample_count <= 0) continue;

            if (whisper_full(ctx, params, pcmf32.data() + sample_start, sample_count) != 0) {
                fprintf(stderr, "Error: Failed to process segment.\n");
                continue;
            }

            const int n_whisper_segments = whisper_full_n_segments(ctx);
            for (int k = 0; k < n_whisper_segments; ++k) {
                accumulated_cleaned.clear();
                char_to_token.clear();
                const int n_tokens = whisper_full_n_tokens(ctx, k);
                
                for (int l = 0; l < n_tokens; ++l) {
                    whisper_token token_id = whisper_full_get_token_id(ctx, k, l);
                    if (token_id >= whisper_token_beg(ctx)) continue;

                    const char * token_text = whisper_full_get_token_text(ctx, k, l);
                    append_cleaned_word(token_text, accumulated_cleaned, char_to_token, l);
                }

                size_t pos = accumulated_cleaned.find(target_word);
                if (pos != std::string::npos) {
                    int token_index = char_to_token[pos];
                    whisper_token_data token_data = whisper_full_get_token_data(ctx, k, token_index);
                    
                    final_start_seconds = t0 + (token_data.t0 * 0.01f);
                    break;
                }
            }
            if (final_start_seconds >= 0) break;
        }

        whisper_vad_free_segments(segments);
        if (final_start_seconds >= 0) break;
    }

    whisper_vad_free(vctx);
    whisper_free(ctx);

    if (final_start_seconds < 0) {
        fprintf(stderr, "Target word '%s' not detected. Not creating an output file.\n", target_word.c_str());
        return 0;
    }

    fprintf(stderr, "Detected target word '%s' at %.3f seconds.\n", target_word.c_str(), final_start_seconds);

    std::string trim_cmd = "ffmpeg -hide_banner -loglevel error -nostdin -y -i \"" + audio_file + "\" -ss " + std::to_string(final_start_seconds) + " -c copy \"" + output_file + "\"";
    fprintf(stderr, "Trimming audio and saving to %s...\n", output_file.c_str());
    if (system(trim_cmd.c_str()) != 0) {
        fprintf(stderr, "Error: Failed to trim audio using ffmpeg.\n");
        return 1;
    }

    fprintf(stderr, "Successfully created %s.\n", output_file.c_str());

    return 0;
}
