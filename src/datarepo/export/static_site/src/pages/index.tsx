import '../styles/global.css'

import { ExternalLinkIcon } from '@radix-ui/react-icons'
import { Button, Flex, Heading, Select, Separator, Theme } from '@radix-ui/themes'
import '@radix-ui/themes/styles.css'
import { Outlet, generatePath, useNavigate, useParams } from 'react-router-dom'
import { useState } from 'react'

import { datarepo } from '../lib/data'
import { FuzzySearchBox } from '../components/FuzzySearchBox'
import MobileMenu, { MobileMenuButton } from '../components/MobileMenu'

export default function RootPage () {
  const { catalogKey } = useParams()
  const navigate = useNavigate()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  return (
    <Theme appearance='dark' accentColor='blue'>
      <Flex direction='column' height='100dvh'>
        {/* Mobile Header */}
        <Flex 
          px='3' 
          py='2' 
          align='center' 
          gap='3' 
          className='mobile-only mobile-stack'
          direction={{ initial: 'column', sm: 'row' }}
        >
          <Flex justify='between' align='center' width='100%'>
            <MobileMenuButton onClick={() => setIsMobileMenuOpen(true)} />
            
            <Flex gap='3' align='center'>
              <Heading as='h1' size='3'>datarepo</Heading>
              
              <Select.Root
                value={catalogKey}
                onValueChange={(catalogKey) => {
                  navigate(generatePath('/:catalogKey', { catalogKey }))
                }}
              >
                <Select.Trigger variant='surface' />
                <Select.Content>
                  {datarepo.catalogs.map((catalog) => (
                    <Select.Item key={catalog.name} value={catalog.name}>
                      {catalog.name}
                    </Select.Item>
                  ))}
                </Select.Content>
              </Select.Root>
            </Flex>
            
            <div style={{ width: '36px' }} /> {/* Spacer to balance the layout */}
          </Flex>

          {catalogKey && (
            <Flex justify='center' className='mobile-full-width' pt='3'>
              <FuzzySearchBox enableKeyboardShortcuts catalogKey={catalogKey} />
            </Flex>
          )}
        </Flex>

        {/* Desktop Header - Original Structure */}
        <Flex px='3' align='center' gap='4' className='desktop-only'>
          <Flex flexGrow='1' py='2' gap='4' align='center'>
            <Heading as='h1' size='3'>datarepo</Heading>

            <Select.Root
              value={catalogKey}
              onValueChange={(catalogKey) => {
                navigate(generatePath('/:catalogKey', { catalogKey }))
              }}
            >
              <Select.Trigger variant='surface' />

              <Select.Content>
                {datarepo.catalogs.map((catalog) => (
                  <Select.Item key={catalog.name} value={catalog.name}>
                    {catalog.name}
                  </Select.Item>
                ))}
              </Select.Content>
            </Select.Root>
          </Flex>

          {catalogKey && <FuzzySearchBox enableKeyboardShortcuts catalogKey={catalogKey} />}

          <Flex flexGrow='1' py='2' justify='end' align='center'>
            <Button
              variant='soft'
              onClick={() => {
                const catalog = datarepo.catalogs.find(c => c.name === catalogKey)
                const jupyterhubUrl = catalog?.metadata?.jupyterhub_url
                if (jupyterhubUrl) {
                  window.open(jupyterhubUrl, '_blank')
                } else {
                  console.warn('No JupyterHub URL found for the selected catalog.')
                }
              }}
            >
              <ExternalLinkIcon /> Run on JupyterHub
            </Button>
          </Flex>
        </Flex>

        <Separator size='4' />

        <Outlet />

        {/* Mobile Menu */}
        <MobileMenu 
          isOpen={isMobileMenuOpen}
          onClose={() => setIsMobileMenuOpen(false)}
        />
      </Flex>
    </Theme>
  )
}
