# -*- coding: utf-8 -*-


import PyPDF4
import optparse
from PyPDF4 import PdfFileReader


def printMeta(filename):
    pdfFile = PdfFileReader(open(filename, 'rb'))
    docInfo = pdfFile.getDocumentInfo()
    print('[*] PDF MetaData For: {}'.format(filename))
    for metaItem in docInfo:
        print('[+] {0} : {1}'.format(metaItem, docInfo[metaItem]))


def main():
    filename = r'attention_20201006_1303'
    filepath_template = r'D:\Users\liyuncong\PycharmProjects\ASO\docs\ASMOTE\paper\figures\%s.pdf'
    # parser = optparse.OptionParser('usage %prog + -F <PDF file name>')
    # parser.add_option('-F', dest='filename', type='string', help='specify PDF file name',
    #                   default=r'D:\Users\liyuncong\PycharmProjects\liyuncong-paper\liyuncong_paper\nlp\sentiment analysis\paper\absa\conference\eacl2021\AGF-ASOTE\figures\attention_20201006_1303.pdf')
    # (options, args) = parser.parse_args()
    # fileName = options.filename
    # if fileName == None:
    #     print(parser.usage)
    #     exit(0)
    # else:
    #     printMeta(fileName)
    fileName = filepath_template % filename
    printMeta(fileName)


if __name__ == '__main__':
    main()
