'''
This extension disables interception from the Proxy and automatically injects a bit of
Javascript and HTML into all HTTP responses passing through Burp.

We add a `<div id="score"></div>` element to the end of the body of all HTTP responses.

We also add an extra line of Javascript to the Agar.io client code to write the current
score to that div on every iteration of the game loop.

Note: The Agar.io client code is obfuscated/minified. When the site updates, the variable
and function names will change. See README.md for the steps to update this script.
'''
from burp import IBurpExtender, IProxyListener

# Eg: `P = Math.max(P, Db());
PREVIOUS_SCORE_VARIABLE_NAME = 'P'
GET_CURRENT_SCORE_FUNCTION_NAME = 'Db'

class BurpExtender(IBurpExtender):

    def registerExtenderCallbacks(self, callbacks):
        print "AgarIO Extension loaded. Waiting for HTTP traffic..."
        helpers = callbacks.getHelpers()
        callbacks.setExtensionName("AgarIO Extension")

        # Disable interception from the Proxy. This stops Burp from freezing
        # all HTTP traffic that passes through it and waiting for the user
        # to click "forward." This plugin will still receive all HTTP messages
        # before they're automatically forwarded.
        callbacks.setProxyInterceptionEnabled(False)

        callbacks.registerProxyListener(ProxyListener(helpers))


class ProxyListener(IProxyListener):
    def __init__(self, helpers):
        self._helpers = helpers

    def processProxyMessage(self, messageIsRequest, message):
        messageIsResponse = not messageIsRequest
        if messageIsResponse:
            print("Intercepting HTTP response")
            messageInfo = message.getMessageInfo()
            rawResponse = messageInfo.getResponse()
            response = self._helpers.analyzeResponse(rawResponse)
            headers = response.getHeaders()
            rawMsgBody = rawResponse[response.getBodyOffset():]
            msgBody = self._helpers.bytesToString(rawMsgBody)

            newBody = self._editAgarHtml(msgBody)
            newBodyBytes = self._helpers.stringToBytes(newBody)

            newResponse = self._helpers.buildHttpMessage(headers, newBodyBytes)
            messageInfo.setResponse(newResponse)
        
        return

    def _editAgarHtml(self, html):
        targetScoreString = '%s=Math.max(%s,%s());' % (
            PREVIOUS_SCORE_VARIABLE_NAME, PREVIOUS_SCORE_VARIABLE_NAME,
            GET_CURRENT_SCORE_FUNCTION_NAME)
        scoreStringAddition = 'document.getElementById("score").innerHTML=%s();' % (
            GET_CURRENT_SCORE_FUNCTION_NAME)
        newScoreString = targetScoreString + scoreStringAddition
        html = html.replace(targetScoreString, newScoreString)

        bodyEndTag = '</body>'
        scoreDiv = '<div id="score"></div>'
        newBodyEndTag = scoreDiv + bodyEndTag
        html = html.replace(bodyEndTag, newBodyEndTag)

        return html
