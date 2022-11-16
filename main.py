import discord
import speak

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents = intents)

### @event ###

@client.event
async def on_ready():
	print(f'{client.user} activated')
speak.speaking(client)

### $Open RUN Close$ ###

f = open('token', 'r')
client.run(f.read())
f.close()