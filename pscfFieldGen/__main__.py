# Library imports
from pscfFieldGen.generation import (
    generate_field_file,
    read_input_file )
from pscfFieldGen.util.tracing import TraceLevel, TRACER
# Standard Library Imports
import argparse
from pathlib import Path

if __name__=="__main__":
    # Get command file from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--file","-f", type=str, required=True)
    trace_group = parser.add_mutually_exclusive_group()
    trace_group.add_argument("--mode","-m",choices=["silent","echo","trace","verbose","debug"])
    trace_group.add_argument("--silent","-s",action='store_true')
    trace_group.add_argument("--echo","-e", action='store_true')
    trace_group.add_argument("--trace","-t", action='store_true')
    trace_group.add_argument("--verbose","-v", action='store_true')
    trace_group.add_argument("--debug","-d", action='store_true')
    args = parser.parse_args()
    filepath = Path(args.file)
    # Determine Trace detail level from mutually exclusive option set
    if args.mode is not None:
        mode = args.mode
        mode = mode.lower()
        if mode == "silent":
            tflag = TraceLevel.NONE
        elif mode == "echo":
            tflag = TraceLevel.ECHO
        elif mode == "trace":
            tflag = TraceLevel.EVENT
        elif mode == "verbose":
            tflag = TraceLevel.ALL
        elif mode == "debug":
            tflag = TraceLevel.DEBUG
        else:
            raise(RuntimeError("Unexpected value for '--mode', {}.".format(mode)))
    elif args.silent:
        tflag = TraceLevel.NONE
    elif args.echo:
        tflag = TraceLevel.ECHO
    elif args.trace:
        tflag = TraceLevel.EVENT
    elif args.verbose:
        tflag = TraceLevel.ALL
    elif args.debug:
        tflag = TraceLevel.DEBUG
    else:
        tflag = TraceLevel.ECHO
    # Trace detail level chosen
    param, calc, out = read_input_file(filepath, trace=tflag)
    
    # Generate File
    TRACER.trace("Generating Field File.",TraceLevel.EVENT)
    generate_field_file(param, calc, out)
    TRACER.trace("Field Generation Complete.",TraceLevel.EVENT)
            
            
            
