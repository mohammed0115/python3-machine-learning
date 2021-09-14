import os
import paramiko
server='212.0.148.112'
username='ashrafcom'
password='Admin@Ashrafcom$morsal.2021##$$'
ssh = paramiko.SSHClient() 
ssh.load_host_keys(os.path.expanduser(os.path.join("~", "/home/mohammed/Documents/import/morsal", "testServer.bin")))
ssh.connect(server, username=username, password=password)
#sftp = ssh.open_sftp()
#sftp.put(localpath, remotepath)
#sftp.close()
#ssh.close()
stdin, stdout, stderr = client.exec_command('ls -l')
print(stdin, stdout, stderr)

