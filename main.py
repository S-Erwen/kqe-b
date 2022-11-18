import discord
import data

d_intents = discord.Intents.default()
d_intents.message_content = True

client = discord.Client(intents = d_intents)

### @event ###

@client.event
async def on_ready():
	print(f'{client.user} activated')
data
data.get_mess(client)

### $Open RUN Close$ ###

f = open('token', 'r')
client.run(f.read())
f.close()