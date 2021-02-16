# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright [2021] Optimal Control Suite
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import time
from datetime import datetime

from OptimizationCode.ServerHandler import ServerHandler as SH


def main():
    # object
    server_obj = SH()
    print("Start Job Handler loop")
    # Loop part
    while server_obj.is_running:
        print(datetime.now())
        # Get incoming jobs
        incoming_jobs_list = server_obj.get_incoming_jobs_list()
        # Check Server Status
        server_obj.check_server_status()
        # Run the accepted jobs jobs
        if not server_obj.is_busy:
            for job in incoming_jobs_list:
                server_obj.run_job(job)
        # Update number current jobs
        server_obj.update_job_number()
        time.sleep(2)


if __name__ == "__main__":
    main()
