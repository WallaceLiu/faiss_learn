(faiss_1.6.0) [matthijs@matthijs-mbp /Users/matthijs/Desktop/faiss_github/issue_1127] lldb python
(lldb) target create "python"
Current executable set to 'python' (x86_64).
(lldb) run faiss_memory_leak.py
Process 70943 launched: '/Users/matthijs/miniconda3/envs/faiss_1.6.0/bin/python' (x86_64)
Loading faiss.
2 locations added to breakpoint 1
python version:3.7.5
faiss version: 1.6.0
Process 70943 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGSTOP
    frame #0: 0x00007fff653b55be libsystem_kernel.dylib`__select + 10
libsystem_kernel.dylib`__select:
->  0x7fff653b55be <+10>: jae    0x7fff653b55c8            ; <+20>
    0x7fff653b55c0 <+12>: movq   %rax, %rdi
    0x7fff653b55c3 <+15>: jmp    0x7fff653ae68d            ; cerror
    0x7fff653b55c8 <+20>: retq
(lldb) b faiss::IndexIVFFlat::~IndexIVFFlat()
Breakpoint 2: 2 locations.
(lldb) c
Process 70943 resuming
  UID   PID  PPID        F CPU PRI NI       SZ    RSS WCHAN     S             ADDR TTY           TIME CMD
  501 70943 70945     5806   0  31  0  4515556  33296 -      SX                  0 ttys015    0:00.24 python faiss_mem
Start memory: None
  UID   PID  PPID        F CPU PRI NI       SZ    RSS WCHAN     S             ADDR TTY           TIME CMD
  501 70943 70945     5806   0  31  0  4555636  73476 -      SX                  0 ttys015    0:00.28 python faiss_mem
first index read: None
Process 70943 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2 2.2
    frame #0: 0x00000001125d37e0 _swigfaiss.so`faiss::IndexIVFFlat::~IndexIVFFlat()
_swigfaiss.so`faiss::IndexIVFFlat::~IndexIVFFlat:
->  0x1125d37e0 <+0>: pushq  %rbp
    0x1125d37e1 <+1>: movq   %rsp, %rbp
    0x1125d37e4 <+4>: pushq  %rbx
    0x1125d37e5 <+5>: pushq  %rax
(lldb) c
Process 70943 resuming
  UID   PID  PPID        F CPU PRI NI       SZ    RSS WCHAN     S             ADDR TTY           TIME CMD
  501 70943 70945     5806   0  31  0  4555636  73504 -      SX                  0 ttys015    0:00.28 python faiss_mem
dereference first index: None
  UID   PID  PPID        F CPU PRI NI       SZ    RSS WCHAN     S             ADDR TTY           TIME CMD
  501 70943 70945     5806   0  31  0  4635636 153508 -      SX                  0 ttys015    0:00.36 python faiss_mem
second index read: None
Process 70943 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = breakpoint 1.2 2.2
    frame #0: 0x00000001125d37e0 _swigfaiss.so`faiss::IndexIVFFlat::~IndexIVFFlat()
_swigfaiss.so`faiss::IndexIVFFlat::~IndexIVFFlat:
->  0x1125d37e0 <+0>: pushq  %rbp
    0x1125d37e1 <+1>: movq   %rsp, %rbp
    0x1125d37e4 <+4>: pushq  %rbx
    0x1125d37e5 <+5>: pushq  %rax
(lldb) c
Process 70943 resuming
  UID   PID  PPID        F CPU PRI NI       SZ    RSS WCHAN     S             ADDR TTY           TIME CMD
  501 70943 70945     5806   0  31  0  4635636 153508 -      SX                  0 ttys015    0:00.37 python faiss_mem
dereference second index: None
  UID   PID  PPID        F CPU PRI NI       SZ    RSS WCHAN     S             ADDR TTY           TIME CMD
  501 70943 70945     5806   0  31  0  4635636 153508 -      SX                  0 ttys015    0:00.37 python faiss_mem
final: None
Process 70943 exited with status = 0 (0x00000000)
(lldb)