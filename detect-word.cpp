#include "whisper.h"
#include "common.h"
#include "common-whisper.h"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <cstring>

// Function to execute a shell command and return the output
std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    auto pipe = popen(cmd, "r");
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    try {
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

std::string clean_word(const std::string & word) {
    std::string cleaned = "";
    for (char c : word) {
        if (std::isalnum(c)) {
            cleaned += std::tolower(c);
        }
    }
    return cleaned;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <audio_file> <word> [--output <output_file>]\n", argv[0]);
        return 1;
    }

    std::string audio_file = argv[1];
    std::string target_word = clean_word(argv[2]);
    std::string output_file = "/tmp/trim-output.opus";

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }

    // Convert input audio to 16kHz mono WAV using ffmpeg
    std::string temp_wav = "/tmp/whisper_temp.wav";
    std::string ffmpeg_cmd = "ffmpeg -hide_banner -loglevel error -y -i \"" + audio_file + "\" -ar 16000 -ac 1 -c:a pcm_s16le " + temp_wav;
    if (system(ffmpeg_cmd.c_str()) != 0) {
        fprintf(stderr, "Error: Failed to convert audio to WAV using ffmpeg.\n");
        return 1;
    }

    // Load audio data
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    if (!read_audio_data(temp_wav, pcmf32, pcmf32s, false)) {
        fprintf(stderr, "Error: Failed to read audio data from %s\n", temp_wav.c_str());
        return 1;
    }

    // Initialize VAD context
    struct whisper_vad_context_params vparams = whisper_vad_default_context_params();
    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params("/home/daniel/archivos/ggml-silero-v6.2.0.bin", vparams);
    if (vctx == nullptr) {
        fprintf(stderr, "Error: Failed to initialize VAD context.\n");
        return 1;
    }

    // Detect speech segments
    struct whisper_vad_params vad_params = whisper_vad_default_params();
    struct whisper_vad_segments * segments = whisper_vad_segments_from_samples(vctx, vad_params, pcmf32.data(), pcmf32.size());
    if (segments == nullptr) {
        fprintf(stderr, "Error: Failed to detect speech segments.\n");
        return 1;
    }

    int n_vad_segments = whisper_vad_segments_n_segments(segments);

    // Initialize whisper context
    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * ctx = whisper_init_from_file_with_params("/home/daniel/archivos/ggml-large-v3-turbo-q5_0.bin", cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "Error: Failed to initialize whisper context.\n");
        return 1;
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    params.beam_search.beam_size = 5;
    params.print_progress = false;
    params.print_special = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.translate = false;
    params.language = "auto";
    params.n_threads = 8;
    params.token_timestamps = true;
    params.no_context = true;
    params.single_segment = false;
    params.suppress_blank = true;
    params.suppress_nst = true;

    float final_start_seconds = -1.0f;

    for (int i = 0; i < n_vad_segments; ++i) {
        float t0 = whisper_vad_segments_get_segment_t0(segments, i) * 0.01f;
        float t1 = whisper_vad_segments_get_segment_t1(segments, i) * 0.01f;
        int sample_start = (int)(t0 * 16000);
        int sample_count = (int)((t1 - t0) * 16000);

        // Ensure we don't go out of bounds
        if (sample_start >= pcmf32.size()) continue;
        if (sample_start + sample_count > pcmf32.size()) {
            sample_count = pcmf32.size() - sample_start;
        }
        if (sample_count <= 0) continue;

        if (whisper_full(ctx, params, pcmf32.data() + sample_start, sample_count) != 0) {
            fprintf(stderr, "Error: Failed to process segment %d.\n", i);
            continue;
        }

        const int n_whisper_segments = whisper_full_n_segments(ctx);
        for (int j = 0; j < n_whisper_segments; ++j) {
            std::string accumulated_cleaned = "";
            std::vector<int> char_to_token;
            const int n_tokens = whisper_full_n_tokens(ctx, j);
            
            for (int k = 0; k < n_tokens; ++k) {
                whisper_token token_id = whisper_full_get_token_id(ctx, j, k);
                if (token_id >= whisper_token_beg(ctx)) continue;

                const char * token_text = whisper_full_get_token_text(ctx, j, k);
                if (token_text == nullptr) continue;
                
                std::string cleaned = clean_word(token_text);
                for (size_t l = 0; l < cleaned.length(); ++l) {
                    char_to_token.push_back(k);
                }
                accumulated_cleaned += cleaned;
            }

            size_t pos = accumulated_cleaned.find(target_word);
            if (pos != std::string::npos) {
                int token_index = char_to_token[pos];
                whisper_token_data token_data = whisper_full_get_token_data(ctx, j, token_index);
                
                // token_data.t0 is relative to the start of this VAD segment
                final_start_seconds = t0 + (token_data.t0 * 0.01f);
                break;
            }
        }
        if (final_start_seconds >= 0) break;
    }

    whisper_vad_free_segments(segments);
    whisper_vad_free(vctx);
    whisper_free(ctx);
    remove(temp_wav.c_str());

    if (final_start_seconds < 0) {
        fprintf(stderr, "Target word '%s' not detected. Not creating an output file.\n", argv[2]);
        return 0;
    }

    fprintf(stderr, "Detected target word '%s' at %.3f seconds.\n", argv[2], final_start_seconds);

    // Trim audio using ffmpeg
    std::string trim_cmd = "ffmpeg -hide_banner -nostdin -y -i \"" + audio_file + "\" -ss " + std::to_string(final_start_seconds) + " -c copy \"" + output_file + "\"";
    fprintf(stderr, "Trimming audio and saving to %s...\n", output_file.c_str());
    if (system(trim_cmd.c_str()) != 0) {
        fprintf(stderr, "Error: Failed to trim audio using ffmpeg.\n");
        return 1;
    }

    fprintf(stderr, "Successfully created %s.\n", output_file.c_str());

    return 0;
}
