import io, os, subprocess, time, sys
from subprocess import PIPE, Popen
from threading  import Thread
from queue import Queue, Empty


class SubProcess(object):
 
  def __init__(self):
    self.process = None
    self.read_queue = None
    self.ON_POSIX = 'posix' in sys.builtin_module_names
    self.TIMEOUT = 0.1 # seconds

  
  # Start a subprocess, and a background thread to read from it.
  def start(self, cmdline):
    #print("SubProcess.start(): %s" % cmdline)

#    proc = subprocess.Popen([cmdline],
    proc = subprocess.Popen(cmdline,
        stdin  = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        bufsize = 1,
        close_fds = self.ON_POSIX)
  
    read_queue = Queue()
    read_thread = Thread(target=self._enqueue_input, args=(proc.stdout, read_queue))
    read_thread.daemon = True
    read_thread.start()

    self.process = proc
    self.read_queue = read_queue
  
    return #proc, read_queue
  
  # Send a command to the process.
  # Carriage return is appended for you.
  # wait_for_reply: if True, will BLOCK until process stops printing, and return the results.
  def write(self, command, wait_for_reply=False):
    # Hack: sleep 5 seconds between commands, so as not to overwhelm lighthouse_console.
    # Do it here as a convenience, to save a lot of typing in the main application.
    time.sleep(5)
  
    # Send the command, followed by carriage return to execute
    # print("> %s" % command)
    self.process.stdin.write(command)
    self.process.stdin.write("\n")
    self.process.stdin.flush()
  
    time.sleep(1) # Delay at least long enough for console to respond
  
    if wait_for_reply:
      reply = self.read(0, self.TIMEOUT)
    else:
      reply = None
  
    return reply


  # Read from the process's STDOUT.
  # Will BLOCK until the process is done printing, unless you specify max_lines or a timeout.
  def read(self, max_lines = 0, timeout = 0): #timeout = self.TIMEOUT):
    reply = ""
    num_lines = 0
  
    while(True):
      if max_lines > 0 and num_lines >= max_lines:
        break
  
      line = self._get_line(timeout or self.TIMEOUT)
      if line:
        line = line.decode()
        reply += line
        num_lines += 1
      else:
        #print("EOF")
        break
  
    return reply

  # Some commands (like "dump") produce mountains of output, which the caller wants to ignore.
  # drain() will empty the input queue (read until EOF), allowing the caller
  # to proceed from a known state.
  #
  # Be sure to first send a command that STOPS output (e.g. disable dumping), or else drain() will never return.
  def drain(self):
    reply = self.read(0, self.TIMEOUT)

  def end(self):
    self.process.terminate()
    
  # Private helper: read from process and queue it to the main thread.
  def _enqueue_input(self, pipe, queue):
    # readline() is a blocking call, but this function runs on a thread
    # so the main application won't block.
    for line in iter(pipe.readline, b''):
      queue.put(line)
    pipe.close()
    # print("_enqueue_input exit")
  

  # Private helper: read one line of text from the process, if available.
  def _get_line(self, timeout):
    line = None
    timeout = timeout or .1
  
    try:
      line = self.read_queue.get(timeout = timeout)
    except:
      pass
  
    return line
  


if __name__ == "__main__":
  print("*** Testing SubProcess module ***")

  console = SubProcess()
  command = ["/usr/bin/curl", "-V"]
  console.start(command)

  # Drain the program's initial output
#  console.drain()


  # Test commands and responses
  while(True):
      r = console.read(max_lines = 10)
      print(r)
      break
    
#    r = console.write("serial LHR-", wait_for_reply = True)
#    print("## %s" % r)
  
#    r = console.write("version", wait_for_reply = True)
#    print("## %s" % r)
  
#    r = console.write("battery", wait_for_reply = True)
#    print("## %s" % r)
  
#    console.write("dis")
#    r = console.read()
#    print("## %s" % r)
  
#    console.write("dump")
#    r = console.read()
#    print("## %s" % r)
  
    # Enable IMU dump.
    # You must disable this and call drain() afterwards
#    console.write("imu")
#    r = console.read(max_lines = 10)
#    print("## %s" % r)
  
    # Disable IMU dump.
    # You must call drain() before proceeding
#    console.write("imu")
#    console.drain()
  
#    r = console.write("dump", wait_for_reply = True)
#    print("## %s" % r)
  
