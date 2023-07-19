import math
import subprocess
import logging
import time
import argparse

def generate_sinusoidal_clients(amplitude, period, duration, interval):
    start_time = time.time()
    while time.time() - start_time < duration:
        elapsed_time = time.time() - start_time
        client_count = int((amplitude / 2) * math.sin(2 * math.pi * elapsed_time / period) + (amplitude / 2))
        launch_stress_tool(client_count)
        time.sleep(interval)

def launch_stress_tool(client_count):
    command = f"cassandra-stress write n=1000000 cl=ONE -log file=write_output.txt -node cassandra -pop dist=UNIFORM\(1..1000000\) -rate threads={client_count} fixed=500/s -mode native cql3 protocolVersion=3"
    print("Command: "+str(command))
    #subprocess.run(command, shell=True)

def main():
    logger = logging.getLogger("main")

    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", dest="verbose", action="count",
                        help="set verbosity level [default: %(default)s]")
    parser.add_argument("-s", "--sinusoid", dest="sinusoid", metavar='A,P',
                        help="set the sinusoidal lambda behavior, that varies with amplitude A on period P minutes around the lambda")
    parser.add_argument("-c", "--constant", dest="constant", action="count",
                        help="set the constant load at lambda rate along total duration", required=False)
    parser.add_argument("-l", "--playlist", dest="playlist", help="Set the playlist for the clients", required=False)

    parser.add_argument('--poisson', dest='poisson', action='store_true')
    parser.add_argument('--no-poisson', dest='poisson', action='store_false')
    parser.add_argument('--logfile', dest='logfile', help="Set logFile name", required=True)
    parser.set_defaults(poisson=True)

    # positional arguments (duration, lambda)
    parser.add_argument("duration", type=float, help="set the duration of the experiment in minutes")
    parser.add_argument("lambda", type=float,
                        help="set the (average) arrival rate of lambda clients/minute or normal level of functioning Rnorm for flash crowd")

    # Process arguments
    args = parser.parse_args()

    if args.verbose and args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG)
        # setup logger
        logger.debug("Enabling debug mode")

    else:
        # setup logger
        logging.basicConfig(filename=args.logfile, filemode='w', format='%(message)s', level=logging.INFO)

        # Parâmetros de configuração
        amplitude = 50  # Amplitude do padrão sinusoidal
        period = 60  # Período do padrão sinusoidal (em segundos)
        duration = 3600  # Duração total do teste (em segundos)
        interval = 5  # Intervalo entre cada alteração no número de clientes (em segundos)



        generate_sinusoidal_clients(amplitude, period, args.duration, interval)

# hook for the main function
if __name__ == '__main__':
    main()

