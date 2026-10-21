#include "uaii/c_api/uaii.h"

#include <stdio.h>
#include <string.h>

static int print_phone_preset(uaii_memory_preset_info* info) {
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  if (uaii_resolve_memory_preset("phone", 0, info) != UAII_OK) {
    fprintf(stderr, "resolve phone: %s\n", uaii_last_error());
    return 1;
  }
  printf("preset %s budget %llu kv %lld sink %lld clamped %d available %llu staging %s threads %d rss %llu\n",
         info->label, (unsigned long long)info->ram_budget_bytes, (long long)info->kv_window,
         (long long)info->kv_sink, info->budget_clamped, (unsigned long long)info->available_bytes,
         info->single_staging ? "x1" : "x2", info->max_threads,
         (unsigned long long)uaii_process_rss_bytes());
  return 0;
}

int main(int argc, char** argv) {
  uaii_memory_preset_info info;
  if (print_phone_preset(&info) != 0) return 1;
  if (argc < 2 || argv[1] == NULL || argv[1][0] == '\0') return 0;

  uaii_session_options opts;
  uaii_session_options_init(&opts);
  uaii_session_options_apply_preset(&opts, &info);
  opts.backend = "auto";

  uaii_session* session = NULL;
  if (uaii_session_create(argv[1], &opts, &session) != UAII_OK) {
    fprintf(stderr, "create: %s\n", uaii_last_error());
    return 1;
  }

  const int64_t prompt[3] = {1, 2, 3};
  int64_t out[16];
  size_t n = 0;
  const uaii_status st = uaii_session_generate(session, prompt, 3, 4, out, 16, &n);
  if (st != UAII_OK) {
    fprintf(stderr, "generate: %s\n", uaii_last_error());
    uaii_session_destroy(session);
    return 1;
  }
  printf("tokens %llu rss %llu\n", (unsigned long long)n,
         (unsigned long long)uaii_process_rss_bytes());
  uaii_session_destroy(session);
  return 0;
}
