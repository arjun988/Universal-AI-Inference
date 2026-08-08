#include "commands/generate.hpp"

#include "uaii/core/log.hpp"
#include "uaii/interfaces/tokenizer.hpp"
#include "uaii/ir/graph.hpp"
#include "uaii/loaders/registry.hpp"
#include "uaii/runtime/sampling.hpp"
#include "uaii/runtime/session.hpp"
#include "uaii/tokenizers/gguf_tokenizer.hpp"
#include "uaii/tokenizers/simple_tokenizer.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#endif

namespace uaii {
namespace cli {
namespace {

void print_generate_usage() {
  std::cout
      << "Usage:\n"
      << "  uaii generate --model <path.gguf> --prompt \"...\" [options]\n"
      << "  uaii generate --demo --prompt \"...\" [options]\n\n"
      << "Options:\n"
      << "  --model <path>           GGUF model (or IR .uaii.json + --tokenizer-gguf)\n"
      << "  --demo                   Use built-in tiny GGUF demo model\n"
      << "  --prompt <text>          Prompt text (required unless --prompt-file)\n"
      << "  --prompt-file <path>     Read prompt from file\n"
      << "  --tokenizer-gguf <path>  Tokenizer source when --model is IR\n"
      << "  --max-new-tokens <n>     Default 64\n"
      << "  --max-context <n>        Session max context (0 = model default)\n"
      << "  --backend <name>         Default cpu\n"
      << "  --temperature <f>        0 = greedy (default); >0 enables sampling\n"
      << "  --top-p <f>              Nucleus sampling (default 1 = off)\n"
      << "  --top-k <n>              Top-k filter (0 = off)\n"
      << "  --repetition-penalty <f> >1 penalizes seen tokens (default 1 = off)\n"
      << "  --seed <u64>             RNG seed (omit for nondeterministic)\n"
      << "  --stop-token-id <id>     Repeatable; stop when emitted\n"
      << "  --stop <text>            Repeatable; encode text → stop token ids\n"
      << "  --system <text>          Optional system prefix\n"
      << "  --json                   Machine-readable JSON on stdout\n"
      << "  --stream                 Emit token lines while generating\n"
      << "  --no-color               (global) quieter logs recommended with --json\n";
}

void print_chat_usage() {
  std::cout
      << "Usage:\n"
      << "  uaii chat --model <path.gguf> --jsonl\n"
      << "  uaii chat --demo --jsonl\n\n"
      << "JSONL protocol (one JSON object per stdin line):\n"
      << "  {\"cmd\":\"generate\",\"id\":\"1\",\"prompt\":\"hi\",\"max_new_tokens\":64,\n"
      << "   \"temperature\":0.8,\"top_p\":0.9,\"top_k\":40,\"repetition_penalty\":1.1,\n"
      << "   \"seed\":42,\"stream\":true}\n"
      << "  {\"cmd\":\"reset\",\"id\":\"2\"}\n"
      << "  {\"cmd\":\"quit\"}\n\n"
      << "Events on stdout:\n"
      << "  {\"event\":\"ready\",\"model\":\"...\"}\n"
      << "  {\"event\":\"token\",\"id\":\"1\",\"token\":3,\"text\":\"hello\"}\n"
      << "  {\"event\":\"done\",\"id\":\"1\",\"text\":\"...\",\"prompt_tokens\":N,\"new_tokens\":M,\"ms\":T}\n"
      << "  {\"event\":\"error\",\"id\":\"1\",\"error\":\"...\"}\n";
}

bool has_flag(const std::vector<std::string>& args, const std::string& flag) {
  for (const auto& a : args) {
    if (a == flag) return true;
  }
  return false;
}

std::string get_opt(const std::vector<std::string>& args, const std::string& key,
                    const std::string& def = {}) {
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) return args[i + 1];
  }
  return def;
}

std::vector<std::int64_t> get_opt_i64_all(const std::vector<std::string>& args,
                                          const std::string& key) {
  std::vector<std::int64_t> out;
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) {
      out.push_back(std::strtoll(args[i + 1].c_str(), nullptr, 10));
    }
  }
  return out;
}

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '\\':
        o += "\\\\";
        break;
      case '"':
        o += "\\\"";
        break;
      case '\n':
        o += "\\n";
        break;
      case '\r':
        o += "\\r";
        break;
      case '\t':
        o += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
          o += buf;
        } else {
          o += c;
        }
    }
  }
  return o;
}

std::string read_file(const std::string& path, Error* err) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    if (err) *err = Error::make(ErrorCode::NotFound, "cannot read " + path);
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (err) *err = Error::success();
  return ss.str();
}

std::string extract_json_string(const std::string& line, const char* key) {
  const std::string pat = std::string("\"") + key + "\"";
  auto pos = line.find(pat);
  if (pos == std::string::npos) return {};
  pos = line.find(':', pos + pat.size());
  if (pos == std::string::npos) return {};
  pos = line.find('"', pos + 1);
  if (pos == std::string::npos) return {};
  ++pos;
  std::string out;
  for (; pos < line.size(); ++pos) {
    char c = line[pos];
    if (c == '\\' && pos + 1 < line.size()) {
      char n = line[++pos];
      if (n == 'n') out += '\n';
      else if (n == 'r') out += '\r';
      else if (n == 't') out += '\t';
      else out += n;
    } else if (c == '"') {
      break;
    } else {
      out += c;
    }
  }
  return out;
}

bool extract_json_bool(const std::string& line, const char* key, bool def) {
  const std::string pat = std::string("\"") + key + "\"";
  auto pos = line.find(pat);
  if (pos == std::string::npos) return def;
  pos = line.find(':', pos + pat.size());
  if (pos == std::string::npos) return def;
  auto t = line.find_first_not_of(" \t", pos + 1);
  if (t == std::string::npos) return def;
  if (line.compare(t, 4, "true") == 0) return true;
  if (line.compare(t, 5, "false") == 0) return false;
  return def;
}

std::int64_t extract_json_i64(const std::string& line, const char* key, std::int64_t def) {
  const std::string pat = std::string("\"") + key + "\"";
  auto pos = line.find(pat);
  if (pos == std::string::npos) return def;
  pos = line.find(':', pos + pat.size());
  if (pos == std::string::npos) return def;
  return std::strtoll(line.c_str() + pos + 1, nullptr, 10);
}

double extract_json_f64(const std::string& line, const char* key, double def) {
  const std::string pat = std::string("\"") + key + "\"";
  auto pos = line.find(pat);
  if (pos == std::string::npos) return def;
  pos = line.find(':', pos + pat.size());
  if (pos == std::string::npos) return def;
  return std::strtod(line.c_str() + pos + 1, nullptr);
}

bool json_has_key(const std::string& line, const char* key) {
  return line.find(std::string("\"") + key + "\"") != std::string::npos;
}

std::vector<std::string> get_opt_all(const std::vector<std::string>& args,
                                     const std::string& key) {
  std::vector<std::string> out;
  for (std::size_t i = 0; i + 1 < args.size(); ++i) {
    if (args[i] == key) out.push_back(args[i + 1]);
  }
  return out;
}

runtime::SampleParams sample_from_args(const std::vector<std::string>& args) {
  runtime::SampleParams s;
  if (has_flag(args, "--temperature") || !get_opt(args, "--temperature").empty()) {
    s.temperature = static_cast<float>(std::strtod(get_opt(args, "--temperature", "0").c_str(),
                                                   nullptr));
  }
  if (!get_opt(args, "--top-p").empty()) {
    s.top_p = static_cast<float>(std::strtod(get_opt(args, "--top-p", "1").c_str(), nullptr));
  }
  if (!get_opt(args, "--top-k").empty()) {
    s.top_k = static_cast<std::int32_t>(
        std::strtol(get_opt(args, "--top-k", "0").c_str(), nullptr, 10));
  }
  if (!get_opt(args, "--repetition-penalty").empty()) {
    s.repetition_penalty = static_cast<float>(
        std::strtod(get_opt(args, "--repetition-penalty", "1").c_str(), nullptr));
  }
  if (!get_opt(args, "--seed").empty()) {
    s.seed = static_cast<std::uint64_t>(
        std::strtoull(get_opt(args, "--seed", "0").c_str(), nullptr, 10));
    s.has_seed = true;
  }
  return s;
}

runtime::SampleParams sample_from_jsonl(const std::string& line) {
  runtime::SampleParams s;
  if (json_has_key(line, "temperature")) {
    s.temperature = static_cast<float>(extract_json_f64(line, "temperature", 0.0));
  }
  if (json_has_key(line, "top_p")) {
    s.top_p = static_cast<float>(extract_json_f64(line, "top_p", 1.0));
  }
  if (json_has_key(line, "top_k")) {
    s.top_k = static_cast<std::int32_t>(extract_json_i64(line, "top_k", 0));
  }
  if (json_has_key(line, "repetition_penalty")) {
    s.repetition_penalty =
        static_cast<float>(extract_json_f64(line, "repetition_penalty", 1.0));
  }
  if (json_has_key(line, "seed")) {
    s.seed = static_cast<std::uint64_t>(extract_json_i64(line, "seed", 0));
    s.has_seed = true;
  }
  return s;
}

struct LoadedGen {
  ir::Graph graph;
  std::unique_ptr<ITokenizer> tok;
  std::string model_path;
  std::string tokenizer_path;
  std::unique_ptr<runtime::Session> session;
  bool ready = false;

  void clear() {
    session.reset();
    tok.reset();
    graph = ir::Graph{};
    ready = false;
  }
};

Error load_generator(const std::string& model_path,
                     const std::string& tokenizer_gguf,
                     const std::string& backend,
                     std::int64_t max_context,
                     LoadedGen* out) {
  if (!out) return Error::make(ErrorCode::InvalidArgument, "out null");
  out->clear();
  out->model_path = model_path;
  out->tokenizer_path = tokenizer_gguf.empty() ? model_path : tokenizer_gguf;

  Error err = loaders::load_model(model_path, &out->graph);
  if (!err.ok()) return err;

  out->tok = tokenizers::load_gguf_tokenizer(out->tokenizer_path, &err);
  if (!out->tok) {
    // Tiny demo / missing tokenizer metadata → simple demo vocab.
    out->tok = std::make_unique<tokenizers::SimpleTokenizer>(
        tokenizers::SimpleTokenizer::demo_vocab());
  }

  runtime::SessionOptions opts;
  opts.validate = true;
  opts.weight_init = runtime::WeightInit::None;
  opts.backend_name = backend.empty() ? "cpu" : backend;
  opts.max_context = max_context;
  const std::string device = (opts.backend_name == "cpu") ? "cpu" : "gpu";
  opts.on_load_progress = [backend = opts.backend_name, device](std::size_t loaded,
                                                                std::size_t total,
                                                                const std::string& tensor) {
    const int pct =
        total == 0 ? 100 : static_cast<int>((loaded * 100) / total);
    std::cout << "{\"event\":\"load_progress\",\"loaded\":" << loaded << ",\"total\":" << total
              << ",\"pct\":" << pct << ",\"tensor\":\"" << json_escape(tensor)
              << "\",\"backend\":\"" << json_escape(backend) << "\",\"device\":\"" << device
              << "\"}\n";
    std::cout.flush();
  };
  out->session = std::make_unique<runtime::Session>();
  err = out->session->create(std::move(out->graph), opts);
  if (!err.ok()) {
    out->clear();
    return err;
  }
  out->ready = true;
  return Error::success();
}

std::string build_prompt(const std::string& system, const std::string& user) {
  // Dashboard may already send a full ChatML transcript — don't wrap twice.
  if (user.find("<|im_start|>") != std::string::npos) {
    if (user.find("<|im_start|>assistant") != std::string::npos) return user;
    std::string out = user;
    if (!out.empty() && out.back() != '\n') out.push_back('\n');
    out += "<|im_start|>assistant\n";
    return out;
  }
  // ChatML used by Qwen2.5-Instruct and similar GGUF chat models.
  std::string out;
  if (!system.empty()) {
    out += "<|im_start|>system\n";
    out += system;
    out += "<|im_end|>\n";
  }
  out += "<|im_start|>user\n";
  out += user;
  out += "<|im_end|>\n";
  out += "<|im_start|>assistant\n";
  return out;
}

Error append_stop_strings(ITokenizer* tok, const std::vector<std::string>& stops,
                          std::vector<std::int64_t>* stop_ids) {
  if (!tok || !stop_ids) return Error::make(ErrorCode::InvalidArgument, "stop encode null");
  for (const auto& s : stops) {
    if (s.empty()) continue;
    std::vector<std::int64_t> ids;
    Error err = tok->encode(s, &ids);
    if (!err.ok()) return err;
    for (auto id : ids) stop_ids->push_back(id);
  }
  return Error::success();
}

Error run_text_generate(LoadedGen* gen,
                        const std::string& prompt,
                        std::int64_t max_new_tokens,
                        const std::vector<std::int64_t>& stop_ids,
                        const runtime::SampleParams& sample,
                        bool stream,
                        bool json_mode,
                        const std::string& req_id,
                        std::string* reply_out,
                        std::size_t* prompt_n,
                        std::size_t* new_n,
                        std::int64_t* ms_out) {
  if (!gen || !gen->ready || !gen->tok || !gen->session) {
    return Error::make(ErrorCode::InvalidArgument, "generator not ready");
  }
  std::vector<std::int64_t> prompt_tokens;
  Error err = gen->tok->encode(prompt, &prompt_tokens);
  if (!err.ok()) return err;
  if (prompt_tokens.empty()) {
    return Error::make(ErrorCode::InvalidArgument, "encode produced empty prompt");
  }
  if (prompt_n) *prompt_n = prompt_tokens.size();

  std::string streamed_text;
  std::vector<std::int64_t> generated;
  const auto t0 = std::chrono::steady_clock::now();

  err = gen->session->generate(
      prompt_tokens, max_new_tokens, &generated, stop_ids,
      [&](std::int64_t token_id) {
        for (const auto s : stop_ids) {
          if (token_id == s) return true;  // omit stop tokens from streamed text
        }
        std::vector<std::int64_t> one{token_id};
        std::string piece;
        Error de = gen->tok->decode(one, &piece);
        if (!de.ok()) piece.clear();
        streamed_text += piece;
        if (stream) {
          if (json_mode) {
            std::cout << "{\"event\":\"token\"";
            if (!req_id.empty()) std::cout << ",\"id\":\"" << json_escape(req_id) << "\"";
            std::cout << ",\"token\":" << token_id << ",\"text\":\"" << json_escape(piece)
                      << "\"}\n";
            std::cout.flush();
          } else {
            std::cout << piece << std::flush;
          }
        }
        return true;
      },
      sample);
  if (!err.ok()) return err;

  std::vector<std::int64_t> new_tokens;
  if (generated.size() > prompt_tokens.size()) {
    new_tokens.assign(generated.begin() + static_cast<std::ptrdiff_t>(prompt_tokens.size()),
                      generated.end());
  }
  if (new_n) *new_n = new_tokens.size();

  std::string text = streamed_text;
  if (text.empty()) {
    err = gen->tok->decode(new_tokens, &text);
    if (!err.ok()) return err;
  }
  if (reply_out) *reply_out = text;
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  if (ms_out) *ms_out = ms;
  return Error::success();
}

}  // namespace

int cmd_generate(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    print_generate_usage();
    return args.empty() ? 1 : 0;
  }

  const bool demo = has_flag(args, "--demo");
  std::string model = get_opt(args, "--model");
  std::string tokenizer_gguf = get_opt(args, "--tokenizer-gguf");
  std::string prompt = get_opt(args, "--prompt");
  const std::string prompt_file = get_opt(args, "--prompt-file");
  const std::string system = get_opt(args, "--system");
  const std::string backend = get_opt(args, "--backend", "cpu");
  const std::int64_t max_new =
      std::strtoll(get_opt(args, "--max-new-tokens", "64").c_str(), nullptr, 10);
  const std::int64_t max_ctx =
      std::strtoll(get_opt(args, "--max-context", "0").c_str(), nullptr, 10);
  auto stop_ids = get_opt_i64_all(args, "--stop-token-id");
  const auto stop_texts = get_opt_all(args, "--stop");
  const runtime::SampleParams sample = sample_from_args(args);
  const bool json_mode = has_flag(args, "--json");
  const bool stream = has_flag(args, "--stream");

  if (!prompt_file.empty()) {
    Error err;
    prompt = read_file(prompt_file, &err);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
  }
  if (prompt.empty()) {
    std::cerr << "--prompt or --prompt-file required\n";
    return 1;
  }
  prompt = build_prompt(system, prompt);

  if (demo) {
    Error err = runtime::materialize_tiny_gguf_demo(&model);
    if (!err.ok()) {
      std::cerr << err.to_string() << '\n';
      return 1;
    }
    if (tokenizer_gguf.empty()) tokenizer_gguf = model;
  }
  if (model.empty()) {
    std::cerr << "--model or --demo required\n";
    return 1;
  }

  LoadedGen gen;
  Error err = load_generator(model, tokenizer_gguf, backend, max_ctx, &gen);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }
  err = append_stop_strings(gen.tok.get(), stop_texts, &stop_ids);
  if (!err.ok()) {
    std::cerr << err.to_string() << '\n';
    return 1;
  }

  std::string reply;
  std::size_t prompt_n = 0;
  std::size_t new_n = 0;
  std::int64_t ms = 0;
  err = run_text_generate(&gen, prompt, max_new, stop_ids, sample, stream, json_mode, {},
                          &reply, &prompt_n, &new_n, &ms);
  if (!err.ok()) {
    if (json_mode) {
      std::cout << "{\"ok\":false,\"error\":\"" << json_escape(err.to_string()) << "\"}\n";
    } else {
      std::cerr << err.to_string() << '\n';
    }
    return 1;
  }

  if (json_mode) {
    if (!stream) {
      // stream already printed tokens; always print final object
    }
    std::cout << "{\"ok\":true,\"text\":\"" << json_escape(reply) << "\",\"model\":\""
              << json_escape(model) << "\",\"prompt_tokens\":" << prompt_n
              << ",\"new_tokens\":" << new_n << ",\"ms\":" << ms << "}\n";
  } else if (!stream) {
    std::cout << reply << '\n';
  } else {
    std::cout << '\n';
  }
  return 0;
}

int cmd_chat(const std::vector<std::string>& args) {
  if (args.empty() || has_flag(args, "--help") || has_flag(args, "-h")) {
    print_chat_usage();
    return args.empty() ? 1 : 0;
  }
  if (!has_flag(args, "--jsonl")) {
    std::cerr << "uaii chat currently requires --jsonl (dashboard / automation mode)\n";
    print_chat_usage();
    return 1;
  }

  // Ensure stdout is not block-buffered so JSON events flush immediately.
  std::cout.setf(std::ios::unitbuf);
  // On Windows the CRT may buffer stdout in binary mode; force line-buffered.
#if defined(_WIN32)
  _setmode(_fileno(stdout), _O_TEXT);
  setvbuf(stdout, nullptr, _IOLBF, 0);
#endif

  const bool demo = has_flag(args, "--demo");
  std::string model = get_opt(args, "--model");
  std::string tokenizer_gguf = get_opt(args, "--tokenizer-gguf");
  const std::string backend = get_opt(args, "--backend", "cpu");
  const std::int64_t max_ctx =
      std::strtoll(get_opt(args, "--max-context", "0").c_str(), nullptr, 10);
  const auto default_stops = get_opt_i64_all(args, "--stop-token-id");

  if (demo) {
    Error err = runtime::materialize_tiny_gguf_demo(&model);
    if (!err.ok()) {
      std::cout << "{\"event\":\"error\",\"error\":\"" << json_escape(err.to_string())
                << "\"}\n";
      return 1;
    }
    if (tokenizer_gguf.empty()) tokenizer_gguf = model;
  }
  if (model.empty()) {
    std::cout << "{\"event\":\"error\",\"error\":\"--model or --demo required\"}\n";
    return 1;
  }

  const std::string device = (backend == "cpu") ? "cpu" : "gpu";
  std::cout << "{\"event\":\"load_start\",\"model\":\"" << json_escape(model)
            << "\",\"backend\":\"" << json_escape(backend) << "\",\"device\":\"" << device
            << "\"}\n";
  std::cout.flush();

  LoadedGen gen;
  Error err = load_generator(model, tokenizer_gguf, backend, max_ctx, &gen);
  if (!err.ok()) {
    std::cout << "{\"event\":\"error\",\"error\":\"" << json_escape(err.to_string()) << "\"}\n";
    return 1;
  }

  // Prefer ChatML / EOS stops when present in the tokenizer vocab.
  auto stops = default_stops;
  if (gen.tok) {
    for (const char* s : {"<|im_end|>", "<|endoftext|>"}) {
      std::vector<std::int64_t> ids;
      if (gen.tok->encode(s, &ids).ok() && ids.size() == 1) {
        const std::int64_t id = ids.front();
        if (std::find(stops.begin(), stops.end(), id) == stops.end()) {
          stops.push_back(id);
        }
      }
    }
  }

  std::cout << "{\"event\":\"ready\",\"model\":\"" << json_escape(model) << "\",\"backend\":\""
            << json_escape(backend) << "\",\"device\":\"" << device << "\"}\n";
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    if (std::cin.bad()) {
      // Unrecoverable I/O error — exit the loop.
      break;
    }
    const std::string cmd = extract_json_string(line, "cmd");
    const std::string id = extract_json_string(line, "id");
    if (cmd == "quit" || cmd == "exit") {
      std::cout << "{\"event\":\"bye\"}\n";
      break;
    }
    if (cmd == "reset") {
      // Reload session to clear KV / planner state.
      err = load_generator(model, tokenizer_gguf, backend, max_ctx, &gen);
      if (!err.ok()) {
        std::cout << "{\"event\":\"error\",\"id\":\"" << json_escape(id)
                  << "\",\"error\":\"" << json_escape(err.to_string()) << "\"}\n";
      } else {
        std::cout << "{\"event\":\"reset\",\"id\":\"" << json_escape(id) << "\"}\n";
      }
      std::cout.flush();
      continue;
    }
    if (cmd != "generate") {
      std::cout << "{\"event\":\"error\",\"id\":\"" << json_escape(id)
                << "\",\"error\":\"unknown cmd\"}\n";
      std::cout.flush();
      continue;
    }

    std::string prompt = extract_json_string(line, "prompt");
    const std::string system = extract_json_string(line, "system");
    prompt = build_prompt(system, prompt);
    const std::int64_t max_new = extract_json_i64(line, "max_new_tokens", 64);
    const bool stream = extract_json_bool(line, "stream", true);
    const runtime::SampleParams sample = sample_from_jsonl(line);

    std::string reply;
    std::size_t prompt_n = 0;
    std::size_t new_n = 0;
    std::int64_t ms = 0;
    err = run_text_generate(&gen, prompt, max_new, stops, sample, stream,
                            /*json_mode=*/true, id, &reply, &prompt_n, &new_n, &ms);
    if (!err.ok()) {
      std::cout << "{\"event\":\"error\",\"id\":\"" << json_escape(id) << "\",\"error\":\""
                << json_escape(err.to_string()) << "\"}\n";
    } else {
      std::cout << "{\"event\":\"done\",\"id\":\"" << json_escape(id) << "\",\"text\":\""
                << json_escape(reply) << "\",\"prompt_tokens\":" << prompt_n
                << ",\"new_tokens\":" << new_n << ",\"ms\":" << ms << "}\n";
    }
    std::cout.flush();
  }
  return 0;
}

}  // namespace cli
}  // namespace uaii
